import pygame
import numpy as np
import os
import imageio

# SC2-Inspired Color Palette (V10)
COLOR_GRASS = (20, 45, 20)
COLOR_BRIDGE = (35, 75, 35)
COLOR_CLIFF = (5, 5, 5)
COLOR_ALLY = (74, 144, 226)
COLOR_ENEMY = (233, 78, 78)
COLOR_PROJECTILE = (255, 204, 0)
COLOR_GRID = (25, 55, 25)
COLOR_BEACON = (0, 255, 255)

def lerp(a, b, t):
    return a * (1 - t) + b * t

class ProductionRenderer:
    """ Unified Production Renderer with locked camera and 60fps output. """
    def __init__(self, headless=True, trails_enabled=True, width=800, height=800, map_dims=(32, 32)):
        self.headless = headless
        self.trails_enabled = trails_enabled
        self.width, self.height = width, height
        self.map_w, self.map_h = map_dims
        # Always set SDL_VIDEODRIVER before pygame.init() to avoid hangs
        if not os.environ.get('SDL_VIDEODRIVER'):
            os.environ['SDL_VIDEODRIVER'] = 'dummy' if headless else ''
        pygame.init()
        self.screen = pygame.Surface((width, height), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        if not self.headless:
            self.display = pygame.display.set_mode((width, height))
            pygame.display.set_caption("TwoBridge Production UI")
        # Locked camera: fixed at center of map, constant zoom
        self.cam_pos = np.array([map_dims[0]/2, map_dims[1]/2])
        self.zoom = 1.5
        self.terrain_surf = pygame.Surface((width, height))
        self.cached_cam, self.cached_zoom = None, None
        self.pos_history = []
        self.trail_len = 8

    def world_to_screen(self, pos, cam_pos=None, zoom=None, z=0):
        """Convert world coordinates to screen coordinates. Uses self.cam_pos and self.zoom if not provided."""
        cam_pos = cam_pos if cam_pos is not None else self.cam_pos
        zoom = zoom if zoom is not None else self.zoom
        rel = pos - cam_pos
        scale = (self.width / self.map_w) * zoom
        sx = int(self.width/2 + rel[0] * scale)
        sy = int(self.height/2 - (rel[1] * 0.7) * scale - z * scale)
        return sx, sy

    def _draw_terrain(self, cam_pos=None, zoom=None):
        """Draw terrain once (cached). Camera-locked: only drawn once."""
        cam_pos = cam_pos if cam_pos is not None else self.cam_pos
        zoom = zoom if zoom is not None else self.zoom
        if np.array_equal(cam_pos, self.cached_cam) and zoom == self.cached_zoom:
            return
        for y in range(self.height):
            s = int(10 * (y / self.height))
            c = (max(0, COLOR_GRASS[0]-s), max(0, COLOR_GRASS[1]-s), max(0, COLOR_GRASS[2]-s))
            pygame.draw.line(self.terrain_surf, c, (0,y), (self.width,y))
        for i in range(0, int(self.map_w)+1, 4):
            x = self.world_to_screen((i, 0), cam_pos, zoom)[0]
            pygame.draw.line(self.terrain_surf, COLOR_GRID, (x,0), (x,self.height))
            y = self.world_to_screen((0, i), cam_pos, zoom)[1]
            pygame.draw.line(self.terrain_surf, COLOR_GRID, (0,y), (self.width,y))
        f1 = self.world_to_screen((0.45*32, 0), cam_pos, zoom, z=-0.5)[0]
        f2 = self.world_to_screen((0.55*32, 0), cam_pos, zoom, z=-0.5)[0]
        pygame.draw.rect(self.terrain_surf, (0,0,0), (f1,0, f2-f1, self.height))
        for yr in [(0.15, 0.35), (0.65, 0.85)]:
            pts = [self.world_to_screen((0.45*32, yr[0]*32), cam_pos, zoom),
                   self.world_to_screen((0.55*32, yr[0]*32), cam_pos, zoom),
                   self.world_to_screen((0.55*32, yr[1]*32), cam_pos, zoom),
                   self.world_to_screen((0.45*32, yr[1]*32), cam_pos, zoom)]
            pygame.draw.polygon(self.terrain_surf, COLOR_BRIDGE, pts)
        self.cached_cam, self.cached_zoom = np.copy(cam_pos), zoom

    def render_frame(self, f1, f2, alpha, cam1=None, cam2=None, zoom1=None, zoom2=None, shake_vec=(0,0)):
        """Render a single frame. Camera-locked: cam1=cam2=self.cam_pos, zoom1=zoom2=self.zoom."""
        # Use locked camera (ignore cam1/cam2/zoom1/zoom2 parameters)
        cp = self.cam_pos.copy() + shake_vec
        zm = self.zoom
        
        self._draw_terrain(cp, zm)
        self.screen.blit(self.terrain_surf, (0,0))
        
        if self.trails_enabled:
            pos = lerp(f1["unit_pos"], f2["unit_pos"], alpha)
            self.pos_history.append(pos.copy())
            if len(self.pos_history) > self.trail_len:
                self.pos_history.pop(0)
            for k, trail in enumerate(self.pos_history):
                f = int(60 * (k+1) / self.trail_len)
                r = max(1, 4 - (self.trail_len - k)//2)
                for i, p in enumerate(trail):
                    if not f2["unit_alive"][i]:
                        continue
                    sx, sy = self.world_to_screen(p, cp, zm)
                    pygame.draw.circle(self.screen, (255, 255, 255, f), (sx, sy), r)
        
        self._draw_dynamic_data(f1, f2, alpha, cp, zm)
        return self.screen

    def _draw_dynamic_data(self, f1, f2, alpha, cp, zm):
        """Draw units, projectiles, beacon."""
        if "beacon_pos" in f2:
            b_p = f2["beacon_pos"]
            sxb, syb = self.world_to_screen(b_p, cp, zm)
            alive = f2["unit_alive"][:5]
            if np.any(alive):
                idx = np.argmin(np.where(alive, np.linalg.norm(f2["unit_pos"][:5] - b_p, axis=1), 999))
                sxu, syu = self.world_to_screen(f2["unit_pos"][idx], cp, zm)
                pygame.draw.line(self.screen, (0, 255, 255, 30), (sxb, syb), (sxu, syu), 1)
            p = (np.sin(alpha * np.pi) * 5)
            for r in range(15+int(p), 5, -5):
                pygame.draw.circle(self.screen, (*COLOR_BEACON, 40), (sxb, syb), r)
            pygame.draw.circle(self.screen, COLOR_BEACON, (sxb, syb), 5)
        
        pos = lerp(f1["unit_pos"], f2["unit_pos"], alpha)
        
        # Draw targeting lines
        for i in range(len(pos)):
            if not f2["unit_alive"][i] or f2["targets"][i] < 0:
                continue
            t_idx = f2["targets"][i]
            if not f2["unit_alive"][t_idx]:
                continue
            p1, p2 = self.world_to_screen(pos[i], cp, zm), self.world_to_screen(pos[t_idx], cp, zm)
            nr = np.linalg.norm(pos[i] - pos[t_idx]) < 7.0 
            color = (130, 180, 255) if i < 5 else (255, 130, 130)
            a, w = (150, 3) if nr else (50, 1)
            pygame.draw.line(self.screen, (*color, a), p1, p2, w)
        
        # Draw units with health bars
        r_map, hp1, hp2 = [12, 10, 18], f1["unit_health"], f2["unit_health"]
        for i in range(len(pos)):
            is_d = f1["unit_alive"][i] and not f2["unit_alive"][i]
            if not f2["unit_alive"][i] and not is_d:
                continue
            sx, sy = self.world_to_screen(pos[i], cp, zm)
            r = int(r_map[f2["unit_types"][i]] * (1.0-alpha if is_d else 1.0))
            pygame.draw.ellipse(self.screen, (0, 0, 0, 120), (sx-r, sy-r*0.3, r*2, r*0.6))
            bx, by = self.world_to_screen(pos[i], cp, zm, z=0.1)
            ua = 255 if not is_d else int(255*(1-alpha))
            pygame.draw.circle(self.screen, (*(COLOR_ALLY if i<5 else COLOR_ENEMY), ua), (bx, by), r)
            pygame.draw.circle(self.screen, (255, 255, 255, int(40*(ua/255))), (bx-r//3, by-r//3), r//2)
            pygame.draw.circle(self.screen, (0, 0, 0, ua), (bx, by), r, 2)
            if hp2[i] < hp1[i] and alpha < 0.5:
                pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), r+2)
            if not is_d:
                ang = -np.pi/2 + (f2["attack_timers"][i]/10.0 * 2 * np.pi)
                px, py = bx+int(np.cos(ang)*(r+3)), by+int(np.sin(ang)*(r+3))
                pygame.draw.circle(self.screen, (255,255,255), (px, py), 2)
                mhp = [100., 45., 200.][f2["unit_types"][i]]
                pygame.draw.rect(self.screen, (40, 40, 40), (bx-15, by-r-8, 30, 3))
                pygame.draw.rect(self.screen, (100, 255, 100), (bx-15, by-r-8, int(30 * hp2[i]/mhp), 3))
                v = (f2["unit_pos"][i] - f1["unit_pos"][i]) * 2.5
                vx, vy = self.world_to_screen(pos[i]+v, cp, zm, z=0.1)
                pygame.draw.line(self.screen, (255, 255, 255, 40), (bx, by), (vx, vy), 1)
        
        # Draw projectiles
        pv, pc, act = f1["proj_pos"], f2["proj_pos"], f2["proj_active"]
        for j in range(len(act)):
            if not act[j]:
                continue
            p1, p2 = self.world_to_screen(pv[j], cp, zm, z=0.3), self.world_to_screen(pc[j], cp, zm, z=0.3)
            pygame.draw.line(self.screen, COLOR_PROJECTILE, p1, p2, 4)
            pygame.draw.circle(self.screen, (255, 255, 255), p2, 2)

    def render_episode(self, trajectory, save_path="demo.gif", interp_steps=4):
        """Render full episode as GIF. Camera-locked: fixed position, 60fps output."""
        frames = []
        
        # Precompute camera: locked at center, constant zoom (no following)
        cam_list = [self.cam_pos.copy()] * len(trajectory)
        zoom_list = [self.zoom] * len(trajectory)
        shake_list = [(0, 0)] * len(trajectory)  # No shake
        
        for i in range(len(trajectory)-1):
            f1, f2 = trajectory[i], trajectory[i+1]
            c1, c2 = cam_list[i], cam_list[i+1]
            z1, z2 = zoom_list[i], zoom_list[i+1]
            s_vec = shake_list[i+1]
            
            for j in range(interp_steps):
                surf = self.render_frame(f1, f2, j/interp_steps, c1, c2, z1, z2, shake_vec=s_vec)
                frames.append(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        
        # Output at 60fps: interp_steps subframes per sim step for smooth interpolation
        target_fps = 60
        imageio.mimsave(save_path, frames, fps=target_fps)
        return frames

    def run_interactive(self, trajectory, interp_steps=4, fps=60):
        """Interactive playback. Camera-locked: fixed position."""
        if self.headless:
            return
        
        running, paused, step_idx = True, False, 0
        cp, zp = self.cam_pos.copy(), self.zoom
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        step_idx = min(step_idx+1, len(trajectory)-2)
                    elif event.key == pygame.K_LEFT:
                        step_idx = max(step_idx-1, 0)
                    elif event.key == pygame.K_q:
                        running = False
            
            if not paused:
                step_idx += 1
                if step_idx >= len(trajectory)-1:
                    step_idx = 0
            
            f1, f2 = trajectory[step_idx], trajectory[min(step_idx+1, len(trajectory)-1)]
            
            for j in range(interp_steps):
                surf = self.render_frame(f1, f2, j / interp_steps, cp, cp, zp, zp)
                self.display.blit(surf, (0, 0))
                pygame.display.flip()
                self.clock.tick(fps)
        
        pygame.quit()

def state_to_frame(state):
    s = state.smax_state
    return {
        "unit_pos": np.array(s.unit_positions),
        "unit_alive": np.array(s.unit_alive, dtype=bool),
        "unit_health": np.array(s.unit_health),
        "unit_types": np.array(s.unit_types),
        "targets": np.array(state.persistent_targets),
        "attack_timers": np.array(state.attack_timers),
        "proj_pos": np.array(state.proj_pos),
        "proj_active": np.array(state.proj_active, dtype=bool),
        "beacon_pos": np.array(state.beacon_pos),
    }
