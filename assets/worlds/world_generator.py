#!/usr/bin/env python3
import argparse
import heapq
import json
import math
import os
import random
import re
from pathlib import Path


# BARN style world generator. CONSTANTS:

# World bounds (from gym_bunker_env.py)
XY_MIN = -4.0
XY_MAX = 10.0
WORLD_SIZE = XY_MAX - XY_MIN  # 14 m
FLOOR_AREA = WORLD_SIZE ** 2   # 196 m²

# LiDAR plane height: mobile_base z=0.22 + lidar body z=0.17 = 0.39 m
LIDAR_PLANE_HEIGHT = 0.39

# Obstacle height must exceed LiDAR plane. Min total height = 0.45 m
MIN_OBSTACLE_HEIGHT = 0.45
MAX_OBSTACLE_HEIGHT = 1.2

# Obstacle size ranges (half-extents)
CYL_RADIUS_RANGE = (0.15, 0.6)
BOX_HALF_EXTENT_RANGE = (0.15, 0.6)

# Max floor coverage → ≥60% free floor
MAX_FLOOR_COVERAGE = 0.40

# Minimum distance from world edges where obstacles can be placed.
PLACEMENT_EDGE_MARGIN = 0.3

# Metrics sampling
N_SAMPLE_POINTS = 50
N_VISIBILITY_RAYS = 8
N_DISPERSION_RAYS = 16
DISPERSION_THRESHOLD = 2.0   # open vs blocked boundary (meters)
MAX_RAY_RANGE = 20.0

# A* for tortuosity
ASTAR_GRID_RES = 0.1   # 0.1 m per cell
N_TORTUOSITY_PATHS = 10

# Obstacle colours (RGBA)
CYLINDER_COLOR = "0 0.6 0.2 1"
BOX_COLOR = "0.6 0.2 0 1"

# ─────────────────── Density profiles ──────────────────────────────────────
# Each profile controls number of obstacles and minimum spacing (safety margin)
# to steer the characteristic dimension and overall density.

DENSITY_PROFILES = {
    'sparse': {
        'num_obstacles': (4, 10),
        'safety_margin': 1.8,        # large min gap → char_dim > 1.5
        'cyl_radius': (0.15, 0.35),
        'box_half': (0.15, 0.35),
    },
    'medium': {
        'num_obstacles': (10, 18),
        'safety_margin': 1.2,         # mid-range gaps
        'cyl_radius': (0.15, 0.5),
        'box_half': (0.15, 0.5),
    },
    'dense': {
        'num_obstacles': (16, 26),
        'safety_margin': 0.7,        # tight packing
        'cyl_radius': (0.15, 0.6),
        'box_half': (0.15, 0.6),
    },
}


# ──────────────────────────── Geometry helpers ─────────────────────────────

def _rot2d(angle):
    """2D rotation matrix."""
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s), (s, c))


def _project_polygon(vertices, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    dots = [v[0] * axis[0] + v[1] * axis[1] for v in vertices]
    return min(dots), max(dots)


def _obb_vertices(cx, cy, hx, hy, yaw):
    """Return 4 corners of an oriented box."""
    R = _rot2d(yaw)
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * hx, sy * hy
        wx = cx + R[0][0] * lx + R[0][1] * ly
        wy = cy + R[1][0] * lx + R[1][1] * ly
        corners.append((wx, wy))
    return corners


def _sat_overlap(verts_a, verts_b, margin=0.0):
    """SAT overlap test for two convex polygons with safety margin."""
    for verts in (verts_a, verts_b):
        n = len(verts)
        for i in range(n):
            edge = (verts[(i + 1) % n][0] - verts[i][0],
                    verts[(i + 1) % n][1] - verts[i][1])
            axis = (-edge[1], edge[0])
            length = math.hypot(axis[0], axis[1])
            if length < 1e-12:
                continue
            axis = (axis[0] / length, axis[1] / length)
            min_a, max_a = _project_polygon(verts_a, axis)
            min_b, max_b = _project_polygon(verts_b, axis)
            if max_a + margin < min_b or max_b + margin < min_a:
                return False
    return True


def _circle_polygon_overlap(cx, cy, r, polygon_verts, margin=0.0):
    """Check if a circle overlaps a convex polygon with safety margin."""
    r_eff = r + margin
    min_dist_sq = float('inf')
    for i in range(len(polygon_verts)):
        ax, ay = polygon_verts[i]
        bx, by = polygon_verts[(i + 1) % len(polygon_verts)]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq))
        px, py = ax + t * dx, ay + t * dy
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    if min_dist_sq < r_eff ** 2:
        return True
    return _point_in_polygon(cx, cy, polygon_verts)


def _point_in_polygon(px, py, verts):
    """Ray-casting point-in-polygon test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _footprint_area(obs):
    """Return the 2D footprint area of an obstacle."""
    if obs['type'] == 'cylinder':
        return math.pi * obs['radius'] ** 2
    else:
        return 4.0 * obs['hx'] * obs['hy']


# ──────────────────────────── Collision checking ───────────────────────────

def check_collision(new_obs, existing, margin):
    """Check if new_obs overlaps any existing obstacle (with safety margin)."""
    for obs in existing:
        if _pair_collides(new_obs, obs, margin):
            return True
    return False


def _pair_collides(a, b, margin):
    """Check collision between two obstacles."""
    if a['type'] == 'cylinder' and b['type'] == 'cylinder':
        dist = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
        return dist < (a['radius'] + b['radius'] + margin)
    elif a['type'] == 'box' and b['type'] == 'box':
        va = _obb_vertices(a['x'], a['y'], a['hx'], a['hy'], a['yaw'])
        vb = _obb_vertices(b['x'], b['y'], b['hx'], b['hy'], b['yaw'])
        return _sat_overlap(va, vb, margin)
    else:
        if a['type'] == 'cylinder':
            cyl, box = a, b
        else:
            cyl, box = b, a
        verts = _obb_vertices(box['x'], box['y'], box['hx'], box['hy'], box['yaw'])
        return _circle_polygon_overlap(cyl['x'], cyl['y'], cyl['radius'], verts, margin)


# ──────────────────────── Obstacle generation ──────────────────────────────

def generate_obstacle(obs_id, rng, profile):
    """Generate a single random obstacle dict using a density profile."""
    obs_type = rng.choice(['cylinder', 'box'])
    x = rng.uniform(XY_MIN + PLACEMENT_EDGE_MARGIN, XY_MAX - PLACEMENT_EDGE_MARGIN)
    y = rng.uniform(XY_MIN + PLACEMENT_EDGE_MARGIN, XY_MAX - PLACEMENT_EDGE_MARGIN)
    height = rng.uniform(MIN_OBSTACLE_HEIGHT, MAX_OBSTACLE_HEIGHT)
    half_h = height / 2.0

    if obs_type == 'cylinder':
        radius = rng.uniform(*profile['cyl_radius'])
        return {
            'id': obs_id, 'type': 'cylinder',
            'x': x, 'y': y, 'z': half_h,
            'radius': radius, 'half_h': half_h,
        }
    else:
        hx = rng.uniform(*profile['box_half'])
        hy = rng.uniform(*profile['box_half'])
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        return {
            'id': obs_id, 'type': 'box',
            'x': x, 'y': y, 'z': half_h,
            'hx': hx, 'hy': hy, 'half_h': half_h,
            'yaw': yaw,
        }


def generate_obstacle_field(seed, profile_name):
    """Generate obstacle field using the given density profile."""
    profile = DENSITY_PROFILES[profile_name]
    rng = random.Random(seed)
    num_obstacles = rng.randint(*profile['num_obstacles'])
    safety_margin = profile['safety_margin']
    obstacles = []
    total_area = 0.0
    max_retries = 300

    for i in range(num_obstacles):
        for _ in range(max_retries):
            obs = generate_obstacle(i, rng, profile)
            area = _footprint_area(obs)
            if (total_area + area) / FLOOR_AREA > MAX_FLOOR_COVERAGE:
                continue
            if not check_collision(obs, obstacles, safety_margin):
                obstacles.append(obs)
                total_area += area
                break

    return obstacles


# ──────────────────────── XML formatting ───────────────────────────────────

def format_obstacle_xml(obs):
    """Format a single obstacle as MuJoCo XML body element."""
    x, y, z = obs['x'], obs['y'], obs['z']
    if obs['type'] == 'cylinder':
        r, hh = obs['radius'], obs['half_h']
        return (
            f'    <body name="obstacle_{obs["id"]}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
            f'      <geom type="cylinder" size="{r:.4f} {hh:.4f}" '
            f'rgba="{CYLINDER_COLOR}" contype="1" conaffinity="1"/>\n'
            f'    </body>\n'
        )
    else:
        hx, hy, hh = obs['hx'], obs['hy'], obs['half_h']
        yaw = obs['yaw']
        return (
            f'    <body name="obstacle_{obs["id"]}" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'euler="0 0 {yaw:.6f}">\n'
            f'      <geom type="box" size="{hx:.4f} {hy:.4f} {hh:.4f}" '
            f'rgba="{BOX_COLOR}" contype="1" conaffinity="1"/>\n'
            f'    </body>\n'
        )


def build_world_xml(template_content, obstacles, seed, output_dir, template_path):
    """Insert obstacles into template XML and fix relative paths."""
    def replace_path(match):
        attr_name = match.group(1)
        original_path = match.group(2)
        if os.path.isabs(original_path):
            return match.group(0)
        abs_target = (template_path.parent / original_path).resolve()
        try:
            new_path = os.path.relpath(abs_target, output_dir)
        except ValueError:
            return match.group(0)
        return f'{attr_name}="{new_path}"'

    content = re.sub(r'(file|meshdir)="([^"]+)"', replace_path, template_content)

    if "</worldbody>" not in content:
        raise ValueError("Template XML missing </worldbody> tag")

    header, footer_raw = content.split("</worldbody>", 1)
    footer = "</worldbody>" + footer_raw

    obs_xml = f'\n    <!-- Generated Obstacles (Seed: {seed}, Count: {len(obstacles)}) -->\n'
    for obs in obstacles:
        obs_xml += format_obstacle_xml(obs)

    return header + obs_xml + footer


# ──────────────────────── Metrics computation ──────────────────────────────

def _dist_point_to_obstacle_surface(px, py, obs):
    """Euclidean distance from point (px, py) to nearest surface of obstacle."""
    if obs['type'] == 'cylinder':
        d_center = math.hypot(px - obs['x'], py - obs['y'])
        return max(0.0, d_center - obs['radius'])
    else:
        dx = px - obs['x']
        dy = py - obs['y']
        c, s = math.cos(-obs['yaw']), math.sin(-obs['yaw'])
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        cx = max(-obs['hx'], min(lx, obs['hx']))
        cy = max(-obs['hy'], min(ly, obs['hy']))
        return math.hypot(lx - cx, ly - cy)


def metric_distance_to_closest(px, py, obstacles):
    """Minimum distance from (px,py) to nearest obstacle surface."""
    if not obstacles:
        return MAX_RAY_RANGE
    return min(_dist_point_to_obstacle_surface(px, py, o) for o in obstacles)


def _ray_cast(px, py, angle, obstacles, max_range=MAX_RAY_RANGE):
    """Cast a ray from (px,py) at angle, return distance to first hit."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    min_t = max_range

    for obs in obstacles:
        if obs['type'] == 'cylinder':
            t = _ray_circle_intersect(px, py, dx, dy, obs['x'], obs['y'], obs['radius'])
        else:
            t = _ray_obb_intersect(px, py, dx, dy, obs)
        if t is not None and 0 < t < min_t:
            min_t = t
    return min_t


def _ray_circle_intersect(ox, oy, dx, dy, cx, cy, r):
    """Ray-circle intersection. Returns t or None."""
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return None


def _ray_obb_intersect(ox, oy, dx, dy, box):
    """Ray-OBB intersection using slab method in local frame."""
    cos_a, sin_a = math.cos(-box['yaw']), math.sin(-box['yaw'])
    rx = ox - box['x']
    ry = oy - box['y']
    lox = cos_a * rx - sin_a * ry
    loy = sin_a * rx + cos_a * ry
    ldx = cos_a * dx - sin_a * dy
    ldy = sin_a * dx + cos_a * dy

    tmin = 0.0
    tmax = MAX_RAY_RANGE

    if abs(ldx) < 1e-12:
        if lox < -box['hx'] or lox > box['hx']:
            return None
    else:
        t1 = (-box['hx'] - lox) / ldx
        t2 = (box['hx'] - lox) / ldx
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None

    if abs(ldy) < 1e-12:
        if loy < -box['hy'] or loy > box['hy']:
            return None
    else:
        t1 = (-box['hy'] - loy) / ldy
        t2 = (box['hy'] - loy) / ldy
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None

    return tmin if tmin > 0 else (tmax if tmax > 0 else None)


def metric_average_visibility(px, py, obstacles):
    """Mean of N_VISIBILITY_RAYS equidistant ray-casts from (px, py)."""
    total = 0.0
    for i in range(N_VISIBILITY_RAYS):
        angle = 2.0 * math.pi * i / N_VISIBILITY_RAYS
        total += _ray_cast(px, py, angle, obstacles)
    return total / N_VISIBILITY_RAYS


def metric_dispersion(px, py, obstacles):
    """Count alternations between open (>threshold) and blocked (<=threshold)."""
    states = []
    for i in range(N_DISPERSION_RAYS):
        angle = 2.0 * math.pi * i / N_DISPERSION_RAYS
        d = _ray_cast(px, py, angle, obstacles)
        states.append('open' if d > DISPERSION_THRESHOLD else 'blocked')

    alternations = 0
    for i in range(N_DISPERSION_RAYS):
        if states[i] != states[(i + 1) % N_DISPERSION_RAYS]:
            alternations += 1
    return alternations


def metric_characteristic_dimension(obstacles):
    """Minimum clearance gap between any two obstacles."""
    if len(obstacles) < 2:
        return WORLD_SIZE

    min_gap = float('inf')
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            gap = _pair_surface_distance(obstacles[i], obstacles[j])
            if gap < min_gap:
                min_gap = gap
    return min_gap


def _pair_surface_distance(a, b):
    """Minimum surface-to-surface distance between two obstacles."""
    if a['type'] == 'cylinder' and b['type'] == 'cylinder':
        d = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
        return max(0.0, d - a['radius'] - b['radius'])
    elif a['type'] == 'cylinder' and b['type'] == 'box':
        return max(0.0, _dist_point_to_obstacle_surface(a['x'], a['y'], b) - a['radius'])
    elif a['type'] == 'box' and b['type'] == 'cylinder':
        return max(0.0, _dist_point_to_obstacle_surface(b['x'], b['y'], a) - b['radius'])
    else:
        va = _obb_vertices(a['x'], a['y'], a['hx'], a['hy'], a['yaw'])
        vb = _obb_vertices(b['x'], b['y'], b['hx'], b['hy'], b['yaw'])
        min_d = float('inf')
        for vx, vy in va:
            d = _dist_point_to_obstacle_surface(vx, vy, b)
            if d < min_d:
                min_d = d
        for vx, vy in vb:
            d = _dist_point_to_obstacle_surface(vx, vy, a)
            if d < min_d:
                min_d = d
        return max(0.0, min_d)


# ──────────────────────── A* path planner  ─────────────────────────────────

def _build_occupancy_grid(obstacles):
    """Build a 2D boolean occupancy grid. True = occupied."""
    n = int(WORLD_SIZE / ASTAR_GRID_RES)
    grid = [[False] * n for _ in range(n)]

    for obs in obstacles:
        if obs['type'] == 'cylinder':
            cx_cell = (obs['x'] - XY_MIN) / ASTAR_GRID_RES
            cy_cell = (obs['y'] - XY_MIN) / ASTAR_GRID_RES
            # Inflate by half the robot width (~0.3m) for realistic path planning
            r_cells = (obs['radius'] + 0.3) / ASTAR_GRID_RES
            r2 = r_cells ** 2
            lo_i = max(0, int(cx_cell - r_cells) - 1)
            hi_i = min(n - 1, int(cx_cell + r_cells) + 1)
            lo_j = max(0, int(cy_cell - r_cells) - 1)
            hi_j = min(n - 1, int(cy_cell + r_cells) + 1)
            for gi in range(lo_i, hi_i + 1):
                for gj in range(lo_j, hi_j + 1):
                    if (gi - cx_cell) ** 2 + (gj - cy_cell) ** 2 <= r2:
                        grid[gi][gj] = True
        else:
            hx = obs['hx'] + 0.3
            hy = obs['hy'] + 0.3
            verts = _obb_vertices(obs['x'], obs['y'], hx, hy, obs['yaw'])
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            lo_i = max(0, int((min(xs) - XY_MIN) / ASTAR_GRID_RES) - 1)
            hi_i = min(n - 1, int((max(xs) - XY_MIN) / ASTAR_GRID_RES) + 1)
            lo_j = max(0, int((min(ys) - XY_MIN) / ASTAR_GRID_RES) - 1)
            hi_j = min(n - 1, int((max(ys) - XY_MIN) / ASTAR_GRID_RES) + 1)
            for gi in range(lo_i, hi_i + 1):
                for gj in range(lo_j, hi_j + 1):
                    wx = XY_MIN + gi * ASTAR_GRID_RES
                    wy = XY_MIN + gj * ASTAR_GRID_RES
                    if _point_in_polygon(wx, wy, verts):
                        grid[gi][gj] = True

    return grid, n


def _astar(grid, n, start, goal):
    """A* on grid. Returns path length (Euclidean) or None if no path."""
    si, sj = start
    gi_end, gj = goal

    if grid[si][sj] or grid[gi_end][gj]:
        return None

    def h(i, j):
        return math.hypot(i - gi_end, j - gj)

    open_set = [(h(si, sj), 0.0, si, sj)]
    g_score = {(si, sj): 0.0}
    came_from = {}

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]
    diag_cost = math.sqrt(2.0)

    while open_set:
        _, g_curr, ci, cj = heapq.heappop(open_set)

        if (ci, cj) == (gi_end, gj):
            path_len = 0.0
            cur = (ci, cj)
            while cur in came_from:
                prev = came_from[cur]
                di = abs(cur[0] - prev[0])
                dj = abs(cur[1] - prev[1])
                path_len += diag_cost if (di + dj == 2) else 1.0
                cur = prev
            return path_len * ASTAR_GRID_RES

        if g_curr > g_score.get((ci, cj), float('inf')):
            continue

        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < n and 0 <= nj < n and not grid[ni][nj]:
                step = diag_cost if (abs(di) + abs(dj) == 2) else 1.0
                tentative = g_curr + step
                if tentative < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)] = tentative
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_set, (tentative + h(ni, nj), tentative, ni, nj))

    return None


def metric_tortuosity(obstacles):
    """Mean arc-chord ratio of A* paths between random collision-free point pairs."""
    grid, n = _build_occupancy_grid(obstacles)
    rng = random.Random(12345)

    free_cells = []
    for i in range(n):
        for j in range(n):
            if not grid[i][j]:
                free_cells.append((i, j))

    if len(free_cells) < 2:
        return 1.0

    ratios = []
    attempts = 0
    max_attempts = N_TORTUOSITY_PATHS * 10

    while len(ratios) < N_TORTUOSITY_PATHS and attempts < max_attempts:
        attempts += 1
        start = rng.choice(free_cells)
        goal = rng.choice(free_cells)

        chord = math.hypot(start[0] - goal[0], start[1] - goal[1]) * ASTAR_GRID_RES
        if chord < 3.0:
            continue

        arc = _astar(grid, n, start, goal)
        if arc is not None and chord > 0:
            ratios.append(arc / chord)

    if not ratios:
        return 1.0

    return sum(ratios) / len(ratios)


# ──────────────────────── Sample free points ───────────────────────────────

def _is_point_free(px, py, obstacles, min_clearance=0.35):
    """Check if a point is collision-free with min_clearance from all obstacles."""
    for obs in obstacles:
        if _dist_point_to_obstacle_surface(px, py, obs) < min_clearance:
            return False
    return True


def sample_free_points(obstacles, n_points=N_SAMPLE_POINTS, seed=99):
    """Sample collision-free points in the world."""
    rng = random.Random(seed)
    points = []
    max_attempts = n_points * 20

    for _ in range(max_attempts):
        if len(points) >= n_points:
            break
        px = rng.uniform(XY_MIN + 0.5, XY_MAX - 0.5)
        py = rng.uniform(XY_MIN + 0.5, XY_MAX - 0.5)
        if _is_point_free(px, py, obstacles):
            points.append((px, py))

    return points


# ──────────────────────── Compute all metrics ──────────────────────────────

def compute_metrics(obstacles):
    """Compute all five BARN metrics for a given obstacle field."""
    sample_pts = sample_free_points(obstacles)

    if not sample_pts:
        return {
            'distance_to_closest_obstacle': MAX_RAY_RANGE,
            'average_visibility': MAX_RAY_RANGE,
            'dispersion': 0.0,
            'characteristic_dimension': WORLD_SIZE,
            'tortuosity': 1.0,
        }

    # Distance to closest obstacle (mean over samples)
    dists = [metric_distance_to_closest(px, py, obstacles) for px, py in sample_pts]
    avg_min_dist = sum(dists) / len(dists)

    # Average visibility (mean over samples)
    visibilities = [metric_average_visibility(px, py, obstacles) for px, py in sample_pts]
    avg_visibility = sum(visibilities) / len(visibilities)

    # Dispersion (mean over samples)
    dispersions = [metric_dispersion(px, py, obstacles) for px, py in sample_pts]
    avg_dispersion = sum(dispersions) / len(dispersions)

    # Characteristic dimension (global metric)
    char_dim = metric_characteristic_dimension(obstacles)

    # Tortuosity (global metric with A*)
    print("    Computing tortuosity (A* paths)...", end=" ", flush=True)
    tort = metric_tortuosity(obstacles)
    print(f"done ({tort:.3f})")

    return {
        'distance_to_closest_obstacle': round(avg_min_dist, 4),
        'average_visibility': round(avg_visibility, 4),
        'dispersion': round(avg_dispersion, 4),
        'characteristic_dimension': round(char_dim, 4),
        'tortuosity': round(tort, 4),
    }


# ──────────────────────── Difficulty classification ────────────────────────

def classify_difficulty(metrics):
    """
    Classify difficulty using ALL metrics.

    Easy (Baseline):
        Characteristic Dimension > 1.5 m AND Average Visibility > 3.5 m
        AND Tortuosity < 1.1 AND Dispersion < 4

    Hard (Expert):
        Characteristic Dimension < 0.8 m OR Average Visibility < 1.5 m
        OR Tortuosity > 1.3 OR Dispersion > 8

    Medium (Transition): everything else.
    """
    char_dim = metrics['characteristic_dimension']
    avg_vis = metrics['average_visibility']
    tort = metrics['tortuosity']
    disp = metrics['dispersion']

    if char_dim < 0.8 or avg_vis < 1.5 or tort > 1.3 or disp > 8:
        return 'hard'
    if char_dim > 1.5 and avg_vis > 3.5 and tort < 1.1 and disp < 4:
        return 'easy'
    return 'medium'


# Which density profile to use when we need a particular difficulty
DIFFICULTY_TO_PROFILE = {
    'easy': 'sparse',
    'medium': 'medium',
    'hard': 'dense',
}

# Output folders
FOLDERS = ['train', 'val', 'test']

# Large prime stride to decorrelate consecutive seeds.
# random.Random(100) and random.Random(101) produce very similar sequences;
# multiplying by a large prime spreads them across the state space.
SEED_STRIDE = 7919


def _spread_seed(base_seed, index):
    """Decorrelate seeds: map sequential index to widely-spaced values."""
    return base_seed + index * SEED_STRIDE


# ──────────────────────── Main generation loop ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="High-Precision World Generator with BARN metrics."
    )
    parser.add_argument("--seed", type=int, default=100,
                        help="Base random seed (default: 100)")
    parser.add_argument("--maps-per-folder", type=int, default=9,
                        help="Number of maps per output folder (default: 9, must be divisible by 3)")
    parser.add_argument("--template", type=str, default="empty.xml",
                        help="Path to template XML file")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # Resolve template path
    template_path = Path(args.template) if os.path.isabs(args.template) else script_dir / args.template
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        return

    with open(template_path, 'r') as f:
        template_content = f.read()

    # Create output directories
    folder_dirs = {}
    for folder in FOLDERS:
        d = script_dir / folder
        d.mkdir(parents=True, exist_ok=True)
        folder_dirs[folder] = d

    # ── Generation strategy ──────────────────────────────────────────────
    # 1. Generate maps_per_folder maps for EACH difficulty (easy/medium/hard).
    #    Each folder gets exactly maps_per_folder/3 of each difficulty.
    #    Total = maps_per_folder * 3 folders = 3 * maps_per_folder worlds.
    # 2. Seeds are spread with a large prime stride so worlds look distinct.
    # 3. Distribute worlds across train/val/test so each folder gets
    #    exactly the same count of easy, medium, and hard maps.

    maps_per_folder = args.maps_per_folder
    n_diffs = 3  # easy, medium, hard
    if maps_per_folder % n_diffs != 0:
        print(f"Error: --maps-per-folder ({maps_per_folder}) must be divisible by 3.")
        return
    per_diff_per_folder = maps_per_folder // n_diffs  # e.g. 9/3 = 3
    total_per_diff = per_diff_per_folder * len(FOLDERS)  # e.g. 3*3 = 9
    total_worlds = total_per_diff * n_diffs  # e.g. 9*3 = 27

    print(f"\nGenerating {total_per_diff} maps × 3 difficulties = {total_worlds} total")
    print(f"Each folder (train/val/test) will get {maps_per_folder} maps "
          f"({per_diff_per_folder} easy + {per_diff_per_folder} medium + {per_diff_per_folder} hard).")
    print(f"Base seed: {args.seed}, stride: {SEED_STRIDE}\n")

    # Phase 1: Generate all worlds, grouped by difficulty
    generated = {'easy': [], 'medium': [], 'hard': []}  # list of (seed, obstacles, metrics)

    for target_diff in ('easy', 'medium', 'hard'):
        profile_name = DIFFICULTY_TO_PROFILE[target_diff]
        count = 0
        idx = 0
        max_idx = 500  # safety limit

        print(f"{'='*60}")
        print(f"  Generating {target_diff.upper()} worlds (profile: {profile_name})")
        print(f"{'='*60}")

        while count < total_per_diff and idx < max_idx:
            seed = _spread_seed(args.seed, idx + {'easy': 0, 'medium': 500, 'hard': 1000}[target_diff])
            idx += 1

            print(f"\n  [Seed {seed}] Generating ({profile_name})...", end=" ", flush=True)
            obstacles = generate_obstacle_field(seed, profile_name)
            print(f"{len(obstacles)} obstacles.")

            print(f"    Computing metrics...")
            metrics = compute_metrics(obstacles)
            actual_diff = classify_difficulty(metrics)

            print(f"    Classification: {actual_diff.upper()} "
                  f"(target: {target_diff.upper()}) "
                  f"{'✓' if actual_diff == target_diff else '✗ skip'}")
            print(f"      char_dim={metrics['characteristic_dimension']:.3f}  "
                  f"vis={metrics['average_visibility']:.3f}  "
                  f"tort={metrics['tortuosity']:.3f}  "
                  f"disp={metrics['dispersion']:.3f}  "
                  f"d_min={metrics['distance_to_closest_obstacle']:.3f}")

            if actual_diff != target_diff:
                continue

            count += 1
            generated[target_diff].append((seed, obstacles, metrics))
            print(f"    ✓ Accepted ({count}/{total_per_diff})")

        if count < total_per_diff:
            print(f"\n  ⚠ Only generated {count}/{total_per_diff} {target_diff} maps.")

    # Phase 2: Distribute worlds across folders with mixed difficulties.
    # For each round i, rotate which difficulty goes to which folder:
    #   Round 0: easy→train, medium→val,  hard→test
    #   Round 1: easy→val,   medium→test, hard→train
    #   Round 2: easy→test,  medium→train, hard→val
    # This ensures each folder gets ≈ equal counts of each difficulty.
    print(f"\n{'='*60}")
    print("  Distributing worlds across folders (rotating mix)")
    print(f"{'='*60}")

    folder_contents = {f: [] for f in FOLDERS}
    diff_order = ['easy', 'medium', 'hard']
    n_folders = len(FOLDERS)

    for diff_idx, diff in enumerate(diff_order):
        for i, world_data in enumerate(generated[diff]):
            # Rotate folder assignment: shift by diff_idx so each difficulty
            # starts in a different folder, then advance folder each round
            folder = FOLDERS[(i + diff_idx) % n_folders]
            folder_contents[folder].append((diff, *world_data))

    # Phase 3: Write files
    print()
    for folder in FOLDERS:
        out_dir = folder_dirs[folder]
        print(f"\n  {folder}/ ({len(folder_contents[folder])} maps):")
        for difficulty, seed, obstacles, metrics in folder_contents[folder]:
            xml_content = build_world_xml(template_content, obstacles, seed, out_dir, template_path)
            xml_path = out_dir / f"world_seed_{seed}.xml"
            with open(xml_path, 'w') as f:
                f.write(xml_content)

            meta = {
                'seed': seed,
                'difficulty': difficulty,
                'density_profile': DIFFICULTY_TO_PROFILE[difficulty],
                'num_obstacles': len(obstacles),
                'metrics': metrics,
                'obstacles': [_obstacle_to_dict(o) for o in obstacles],
            }
            meta_path = out_dir / f"world_seed_{seed}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"    {xml_path.name}  [{difficulty:6s}]  "
                  f"cd={metrics['characteristic_dimension']:.2f}  "
                  f"vis={metrics['average_visibility']:.1f}  "
                  f"tort={metrics['tortuosity']:.3f}  "
                  f"disp={metrics['dispersion']:.1f}")

    # Summary
    total_written = sum(len(v) for v in folder_contents.values())
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    for folder in FOLDERS:
        diffs = [d for d, _, _, _ in folder_contents[folder]]
        n_easy = diffs.count('easy')
        n_med = diffs.count('medium')
        n_hard = diffs.count('hard')
        print(f"  {folder:>5s}/: {len(diffs)} maps "
              f"(easy={n_easy}, medium={n_med}, hard={n_hard})")
    print(f"  {'TOTAL':>5s}: {total_written} maps\n")


def _obstacle_to_dict(obs):
    """Serialise obstacle to JSON-friendly dict."""
    d = {
        'type': obs['type'],
        'pos': [round(obs['x'], 4), round(obs['y'], 4), round(obs['z'], 4)],
    }
    if obs['type'] == 'cylinder':
        d['size'] = [round(obs['radius'], 4), round(obs['half_h'], 4)]
    else:
        d['size'] = [round(obs['hx'], 4), round(obs['hy'], 4), round(obs['half_h'], 4)]
        d['yaw'] = round(obs['yaw'], 6)
    return d


if __name__ == "__main__":
    main()
