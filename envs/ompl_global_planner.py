#!/usr/bin/env python3

import numpy as np
import mujoco
import os
from ompl import base as ob
from ompl import geometric as og
from contextlib import contextmanager


def robot_geom_ids_by_subtree(model: mujoco.MjModel, base_body_name: str = "mobile_base") -> set[int]:
    """
    Return all geom ids in the subtree rooted at `base_body_name`.
    Fallback: subtree of the first FREE joint's body.
    """
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)

    children = [[] for _ in range(model.nbody)]
    # Body 0 is the world; iterate from 1..nbody-1
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p >= 0:
            children[p].append(b)

    geoms = set()
    stack = [base_bid]
    while stack:
        b = stack.pop()
        # collect this body's geoms (contiguous range)
        start = model.body_geomadr[b]
        count = model.body_geomnum[b]
        for gid in range(start, start + count):
            geoms.add(gid)
        # traverse its children
        stack.extend(children[b])
    return geoms

def _names_to_geom_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> set[int]:
    out = set()
    for nm in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid != -1:
            out.add(gid)
    return out

@contextmanager
def inflated_margins(model: mujoco.MjModel, inflation: float, exclude_geom_ids: set[int] | None = None):
    """
    Temporarily raise geom margins to at least `inflation` (in meters) for all geoms
    except those listed in `exclude_geom_ids` or named in `exclude_geom_names`.
    """
    exclude_geom_names = ("floor",)
    original = model.geom_margin.copy()
    try:
        excluded = set(exclude_geom_ids or set())
        excluded |= _names_to_geom_ids(model, exclude_geom_names)

        # boolean mask: True => inflate
        mask = np.ones(model.ngeom, dtype=bool)
        if excluded:
            idx = np.array(list(excluded), dtype=int)
            mask[idx] = False

        # increase (never decrease) margins in one go
        mm = model.geom_margin
        mm[mask] = np.maximum(mm[mask], float(inflation))
        yield
    finally:
        model.geom_margin[:] = original



class MujocoStateValidator(ob.StateValidityChecker):

    def __init__(self, si: ob.SpaceInformation, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__(si)
        self.model = model
        self.data = data
        self.base_z = self.model.qpos0[2]

    def isValid(self, state: ob.State) -> bool:
        # Accesores SE2StateSpace
        x, y, yaw = state.getX(), state.getY(), state.getYaw()

        # Save original robot position
        original_qpos = self.data.qpos.copy()

        # Rellena qpos
        qpos = np.zeros(self.model.nq)
        qpos[0:3] = (x, y, self.base_z)  # x, y, z
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, yaw]), b"xyz")
        qpos[3:7] = quat  # orientación base en quaternion

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        for i in range(self.data.ncon):
            c = self.data.contact[i]

            # Get names of the geoms that are colliding
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)

            if name1 != "floor" and name2 != "floor":
                # Restore original position before returning
                self.data.qpos[:] = original_qpos
                mujoco.mj_forward(self.model, self.data)
                return False

            # print(f"Contact: {name1} with {name2}")

        # print("Number of contacts: ", self.data.ncon)

        # Restore original position before returning
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        return True

class OMPLGlobalPlanner:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, bounds: dict, base_body_name: str = "mobile_base"):
        self.model = model
        self.data = data
        self.robot_geom_ids = robot_geom_ids_by_subtree(model, base_body_name)
        self.ss = self.create_simple_setup(model, data, bounds)

    def create_simple_setup(self, model: mujoco.MjModel, data: mujoco.MjData, bounds: dict) -> og.SimpleSetup:
        space = ob.SE2StateSpace()

        rvb = ob.RealVectorBounds(2)
        rvb.setLow(0, bounds["xmin"])
        rvb.setHigh(0, bounds["xmax"])
        rvb.setLow(1, bounds["ymin"])
        rvb.setHigh(1, bounds["ymax"])
        space.setBounds(rvb)

        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(MujocoStateValidator(ss.getSpaceInformation(), model, data))
        return ss

    def plan_path(self, start_pose: tuple, goal_pose: tuple, interpolation_steps: int = 200, inflation_radius: float = 0.25, solve_seconds: float = 30.0):
        print(f"Planning path from {start_pose} to {goal_pose}")

        # Start state
        start = ob.State(self.ss.getStateSpace())
        start().setX(float(start_pose[0]))
        start().setY(float(start_pose[1]))
        start().setYaw(float(start_pose[2]))

        # Goal state
        goal = ob.State(self.ss.getStateSpace())
        goal().setX(float(goal_pose[0]))
        goal().setY(float(goal_pose[1]))
        goal().setYaw(float(goal_pose[2]))

        self.ss.setStartAndGoalStates(start, goal, threshold=0.1)

        # Set the planner
        # planner = og.RRTConnect(self.ss.getSpaceInformation())
        # planner = og.RRT(self.ss.getSpaceInformation())
        planner = og.RRTstar(self.ss.getSpaceInformation())
        # planner = og.InformedRRTstar(self.ss.getSpaceInformation())

        self.ss.setPlanner(planner)

        # Inflate ONLY while planning, then auto-restore
        with inflated_margins(self.model, inflation=inflation_radius, exclude_geom_ids=self.robot_geom_ids):
            if not self.ss.solve(solve_seconds):
                return None
            self.ss.simplifySolution(1.0)

        # margins are restored here
        path = self.ss.getSolutionPath()
        path.interpolate(interpolation_steps) # Here we can set the number of waypoints using path.interpolate(n)


        # Waypoints with corrected yaw angles pointing toward the goal
        wps = []
        for i in range(path.getStateCount()):
            st = path.getState(i)
            x, y = st.getX(), st.getY()

            # Calculate yaw angle pointing toward the next waypoint or goal
            if i < path.getStateCount() - 1:
                nxt = path.getState(i + 1)
                yaw = np.arctan2(nxt.getY() - y, nxt.getX() - x)
            else:
                # Last waypoint: point toward final goal
                yaw = np.arctan2(goal_pose[1] - y, goal_pose[0] - x)
            wps.append((x, y, yaw))
        return wps


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path  = os.path.join(root, "assets", "worlds", "cylinders.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    bounds = dict(xmin=-4.0, xmax=10.0, ymin=-4.0, ymax=10.0)
    start_pose = (-4.0, -4.0, 0.0)  # (x, y, yaw)
    goal_pose = (10.0, 10.0, 0.0)

    planner = OMPLGlobalPlanner(model, data, bounds)

    print("OMPL planning…")
    waypoints = planner.plan_path(start_pose, goal_pose, interpolation_steps=20)

    for waypoint in waypoints:
        print(f"Waypoint: {waypoint}")

    print(f"{len(waypoints)} generated waypoints")


if __name__ == "__main__":
    main()
