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
    """ Unified Production Renderer with temporal stability fixes. """
    def __init__(self, headless=True, trails_enabled=True, width=800, height=800, map_dims=(32, 32)):
        self.headless = headless
        self.trails_enabled = trails_enabled
        self.width, self.height = width, height
        self.map_w, self.map_h = map_dims
        if self.headless: os.environ['SDL_VIDEODRIVER'] = 'dummy'
        else: os.environ.pop('SDL_VIDEODRIVER', None)
        pygame.init()
        self.screen = pygame.Surface((width, height), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        if not self.headless:
            self.display = pygame.display.set_mode((width, height))
            pygame.display.set_caption("TwoBridge Production UI")
        self.cam_pos = np.array([map_dims[0]/2, map_dims[1]/2])
        self.zoom = 1.2
        self.terrain_surf = pygame.Surface((width, height))
        self.cached_cam, self.cached_zoom = None, None
        self.pos_history = []
        self.trail_len = 8

    def world_to_screen(self, pos, cam_pos, zoom, z=0):
        rel = pos - cam_pos
        scale = (self.width / self.map_w) * zoom
        sx = int(self.width/2 + rel[0] * scale)
        sy = int(self.height/2 - (rel[1] * 0.7) * scale - z * scale)
        return sx, sy

    def _draw_terrain(self, cam_pos, zoom):
        if np.array_equal(cam_pos, self.cached_cam) and zoom == self.cached_zoom: return
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

    def render_frame(self, f1, f2, alpha, cam1, cam2, zoom1, zoom2, shake_vec=(0,0)):
        # Apply stable shake vector (calculated once per sim-step or externally)
        cp = lerp(cam1, cam2, alpha) + shake_vec
        zm = lerp(zoom1, zoom2, alpha)
        self._draw_terrain(cp, zm)
        self.screen.blit(self.terrain_surf, (0,0))
        if self.trails_enabled:
            pos = lerp(f1["unit_pos"], f2["unit_pos"], alpha)
            self.pos_history.append(pos.copy())
            if len(self.pos_history) > self.trail_len: self.pos_history.pop(0)
            for k, trail in enumerate(self.pos_history):
                f = int(60 * (k+1) / self.trail_len)
                r = max(1, 4 - (self.trail_len - k)//2)
                for i, p in enumerate(trail):
                    if not f2["unit_alive"][i]: continue
                    sx, sy = self.world_to_screen(p, cp, zm)
                    pygame.draw.circle(self.screen, (255, 255, 255, f), (sx, sy), r)
        self._draw_dynamic_data(f1, f2, alpha, cp, zm)
        return self.screen

    def _draw_dynamic_data(self, f1, f2, alpha, cp, zm):
        if "beacon_pos" in f2:
            b_p = f2["beacon_pos"]
            sxb, syb = self.world_to_screen(b_p, cp, zm)
            alive = f2["unit_alive"][:5]
            if np.any(alive):
                idx = np.argmin(np.where(alive, np.linalg.norm(f2["unit_pos"][:5] - b_p, axis=1), 999))
                sxu, syu = self.world_to_screen(f2["unit_pos"][idx], cp, zm)
                pygame.draw.line(self.screen, (0, 255, 255, 30), (sxb, syb), (sxu, syu), 1)
            p = (np.sin(alpha * np.pi) * 5)
            for r in range(15+int(p), 5, -5): pygame.draw.circle(self.screen, (*COLOR_BEACON, 40), (sxb, syb), r)
            pygame.draw.circle(self.screen, COLOR_BEACON, (sxb, syb), 5)
        pos = lerp(f1["unit_pos"], f2["unit_pos"], alpha)
        for i in range(len(pos)):
            if not f2["unit_alive"][i] or f2["targets"][i] < 0: continue
            t_idx = f2["targets"][i]
            if not f2["unit_alive"][t_idx]: continue
            p1, p2 = self.world_to_screen(pos[i], cp, zm), self.world_to_screen(pos[t_idx], cp, zm)
            nr = np.linalg.norm(pos[i] - pos[t_idx]) < 7.0 
            color = (130, 180, 255) if i < 5 else (255, 130, 130)
            a, w = (150, 3) if nr else (50, 1)
            pygame.draw.line(self.screen, (*color, a), p1, p2, w)
        r_map, hp1, hp2 = [12, 10, 18], f1["unit_health"], f2["unit_health"]
        for i in range(len(pos)):
            is_d = f1["unit_alive"][i] and not f2["unit_alive"][i]
            if not f2["unit_alive"][i] and not is_d: continue
            sx, sy = self.world_to_screen(pos[i], cp, zm)
            r = int(r_map[f2["unit_types"][i]] * (1.0-alpha if is_d else 1.0))
            pygame.draw.ellipse(self.screen, (0, 0, 0, 120), (sx-r, sy-r*0.3, r*2, r*0.6))
            bx, by = self.world_to_screen(pos[i], cp, zm, z=0.1)
            ua = 255 if not is_d else int(255*(1-alpha))
            pygame.draw.circle(self.screen, (*(COLOR_ALLY if i<5 else COLOR_ENEMY), ua), (bx, by), r)
            pygame.draw.circle(self.screen, (255, 255, 255, int(40*(ua/255))), (bx-r//3, by-r//3), r//2)
            pygame.draw.circle(self.screen, (0, 0, 0, ua), (bx, by), r, 2)
            if hp2[i] < hp1[i] and alpha < 0.5: pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), r+2)
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
        pv, pc, act = f1["proj_pos"], f2["proj_pos"], f2["proj_active"]
        for j in range(len(act)):
            if not act[j]: continue
            p1, p2 = self.world_to_screen(pv[j], cp, zm, z=0.3), self.world_to_screen(pc[j], cp, zm, z=0.3)
            pygame.draw.line(self.screen, COLOR_PROJECTILE, p1, p2, 4)
            pygame.draw.circle(self.screen, (255, 255, 255), p2, 2)

    def render_episode(self, trajectory, save_path="demo.gif", interp_steps=4):
        frames = []
        cam_list, zoom_list, shake_list = [], [], []
        c, z = self.cam_pos.copy(), 1.2
        for i, f in enumerate(trajectory):
            alive, eng = f["unit_alive"][:5], np.any(f["targets"] >= 0)
            if np.any(alive):
                t = np.mean(f["unit_pos"][:5][alive], axis=0)
                # Smoother camera: exponential follow instead of clip
                c = c * 0.88 + t * 0.12
                s = np.max(np.linalg.norm(f["unit_pos"][:5][alive] - t, axis=1)) if len(alive)>1 else 0
                bz = np.clip(0.4 * 32 / max(4.0, s), 0.8, 2.0)
                if eng: bz *= 1.15
                z = z * 0.9 + bz * 0.1
            cam_list.append(c.copy()); zoom_list.append(z)
            
            # Smooth shake: damped sine instead of random noise
            shake_phase = i * 0.8
            s_mag = 0.4 if eng else 0.0
            s_vec = np.array([np.sin(shake_phase) * s_mag, 
                              np.cos(shake_phase * 1.3) * s_mag * 0.5])
            shake_list.append(s_vec)
            
        for i in range(len(trajectory)-1):
            f1, f2 = trajectory[i], trajectory[i+1]
            c1, c2, z1, z2 = cam_list[i], cam_list[i+1], zoom_list[i], zoom_list[i+1]
            s_vec = shake_list[i+1]
            for j in range(interp_steps):
                surf = self.render_frame(f1, f2, j/interp_steps, c1, c2, z1, z2, shake_vec=s_vec)
                frames.append(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        imageio.mimsave(save_path, frames, fps=15 * interp_steps)
        return frames

    def run_interactive(self, trajectory, interp_steps=4, fps=60):
        if self.headless: return
        running, paused, step_idx, follow_id, slow_mo = True, False, 0, None, False
        cp, zp = self.cam_pos.copy(), self.zoom
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: paused = not paused
                    if event.key == pygame.K_RIGHT: step_idx = min(step_idx+1, len(trajectory)-2)
                    if event.key == pygame.K_LEFT: step_idx = max(step_idx-1, 0)
                    if event.key == pygame.K_UP: self.zoom *= 1.1
                    if event.key == pygame.K_DOWN: self.zoom /= 1.1
                    if event.key == pygame.K_s: slow_mo = not slow_mo
                    if event.key == pygame.K_f:
                        alive = trajectory[step_idx]["unit_alive"][:5]
                        follow_id = np.where(alive)[0][0] if np.any(alive) and follow_id is None else None
            if not paused:
                step_idx += 1
                if step_idx >= len(trajectory)-1: 
                    step_idx = 0
                    # Reset camera to avoid teleport snap on loop
                    a0 = trajectory[0]["unit_alive"][:5]
                    if np.any(a0):
                        cp = np.mean(trajectory[0]["unit_pos"][:5][a0], axis=0)
            f1, f2 = trajectory[step_idx], trajectory[min(step_idx+1, len(trajectory)-1)]
            eng = np.any(f2["targets"] >= 0)
            if follow_id is not None: 
                ct_target, zt = f2["unit_pos"][follow_id], self.zoom
            else:
                a = f2["unit_alive"][:5]
                if np.any(a):
                    ct_target = np.mean(f2["unit_pos"][:5][a], axis=0)
                    s = np.max(np.linalg.norm(f2["unit_pos"][:5][a] - ct_target, axis=1)) if np.sum(a)>1 else 0
                    zt = np.clip(0.4 * 32 / max(4.0, s), 0.8, 2.0)
                    if eng: zt *= 1.2
                else: ct_target, zt = self.cam_pos, self.zoom
            
            # Apply cinematic exponential follow in interactive mode too
            ct = cp * 0.88 + ct_target * 0.12

            # Calc organic shake
            shake_phase = step_idx * 0.8
            s_mag = 0.4 if eng else 0.0
            s_vec = np.array([np.sin(shake_phase) * s_mag, 
                              np.cos(shake_phase * 1.3) * s_mag * 0.5])
            for j in range(interp_steps):
                surf = self.render_frame(f1, f2, j / interp_steps, cp, ct, zp, zt, shake_vec=s_vec)
                self.display.blit(surf, (0, 0))
                pygame.display.flip()
                self.clock.tick(fps if not slow_mo else fps//4)
            cp, zp = ct, zp * 0.9 + zt * 0.1
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
