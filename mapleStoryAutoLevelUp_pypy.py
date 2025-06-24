'''
PyPy 優化版本的 MapleStory 自動練功程式
執行方式: pypy3 mapleStoryAutoLevelUp_pypy.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import glob
import sys
import pyautogui
import pygetwindow as gw

# Library import
import numpy as np
import cv2
import yaml

# 性能監控
import psutil
import threading

# Local import
from logger import logger
from util import find_pattern_sqdiff, draw_rectangle, screenshot, nms, \
                load_image, get_mask, get_minimap_loc_size, get_player_location_on_minimap, \
                is_mac, nms_matches, override_cfg, load_yaml, get_all_other_player_locations_on_minimap
from KeyBoardController import KeyBoardController
if is_mac():
    from GameWindowCapturorForMac import GameWindowCapturor
else:
    from GameWindowCapturor import GameWindowCapturor
from HealthMonitor import HealthMonitor

class PerformanceMonitor:
    """性能監控器 - PyPy 優化版本"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_history = []
        self.memory_history = []
        self.fps_history = []
        self.start_time = time.time()
        
        # 啟動監控線程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """監控循環"""
        while True:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_mb)
                
                # 保持歷史記錄在合理範圍內
                if len(self.cpu_history) > 100:
                    self.cpu_history.pop(0)
                    self.memory_history.pop(0)
                
                time.sleep(5)  # 每5秒監控一次
                
            except Exception as e:
                logger.error(f"性能監控錯誤: {e}")
                time.sleep(10)
    
    def get_stats(self):
        """獲取性能統計"""
        if not self.cpu_history:
            return {"cpu": 0, "memory": 0, "uptime": 0}
        
        return {
            "cpu": sum(self.cpu_history) / len(self.cpu_history),
            "memory": sum(self.memory_history) / len(self.memory_history),
            "uptime": time.time() - self.start_time
        }
    
    def log_performance(self):
        """記錄性能資訊"""
        stats = self.get_stats()
        logger.info(f"性能統計 - CPU: {stats['cpu']:.1f}%, "
                   f"記憶體: {stats['memory']:.1f}MB, "
                   f"運行時間: {stats['uptime']:.0f}秒")

class MapleStoryBotPyPy:
    '''
    PyPy 優化版本的 MapleStoryBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryBotPyPy
        '''
        # 初始化性能監控
        self.performance_monitor = PerformanceMonitor()
        
        # self.cfg = Config # Configuration
        self.args = args # User arguments
        self.status = "hunting" # 'resting', 'finding_rune', 'near_rune'
        self.idx_routes = 0 # Index of route map
        self.monster_info = [] # monster information
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.rune_detect_level = 0 # higher the level, lower the rune detect threshold
        
        # 性能優化：預分配記憶體
        self.frame_buffer = None
        self.img_frame_buffer = None
        self.img_frame_gray_buffer = None
        
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_switch_status = time.time() # Last status switches timer
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_patrol_last_attack = time.time() # Last patrol attack timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_rune_trigger = time.time() # Last time trigger rune
        
        # Patrol mode
        self.is_patrol_to_left = True # Patrol direction flag
        self.patrol_turn_point_cnt = 0 # Patrol tuning back counter

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))

        # Parse color code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }

        # Set status to hunting for startup
        self.switch_status("hunting")

        if args.patrol:
            # Patrol mode doesn't need map or route
            self.img_map = None
            self.img_routes = []
            self.img_route_rest = None
        else:
            # Load map.png from minimaps/
            self.img_map = load_image(f"minimaps/{args.map}/map.png",
                                      cv2.IMREAD_COLOR)
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{args.map}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = [
                cv2.cvtColor(load_image(p), cv2.COLOR_BGR2RGB) for p in route_files
            ]
            # Load route_rest.png from minimaps/
            self.img_route_rest = cv2.cvtColor(
                load_image(f"minimaps/{args.map}/route_rest.png"), cv2.COLOR_BGR2RGB)

        # Load player's name tag
        self.img_nametag = load_image(f"nametag/{args.nametag}.png")
        self.img_nametag_gray = load_image(f"nametag/{args.nametag}.png", cv2.IMREAD_GRAYSCALE)

        # Load rune images from rune/
        self.img_rune_warning = load_image("rune/rune_warning.png", cv2.IMREAD_GRAYSCALE)
        self.img_runes = [load_image("rune/rune_1.png"),
                          load_image("rune/rune_2.png"),
                          load_image("rune/rune_3.png"),]
        self.img_arrows = {
            "left":
                [load_image("rune/arrow_left_1.png"),
                load_image("rune/arrow_left_2.png"),
                load_image("rune/arrow_left_3.png"),],
            "right":
                [load_image("rune/arrow_right_1.png"),
                load_image("rune/arrow_right_2.png"),
                load_image("rune/arrow_right_3.png"),],
            "up":
                [load_image("rune/arrow_up_1.png"),
                load_image("rune/arrow_up_2.png"),
                load_image("rune/arrow_up_3.png")],
            "down":
                [load_image("rune/arrow_down_1.png"),
                load_image("rune/arrow_down_2.png"),
                load_image("rune/arrow_down_3.png"),],
        }

        # Load monsters images from monster/{monster_name}
        self.monsters = {}
        for monster_name in args.monsters.split(","):
            imgs = []
            for file in glob.glob(f"monster/{monster_name}/{monster_name}*.png"):
                # Add original image
                img = load_image(file)
                imgs.append((img, get_mask(img, (0, 255, 0))))
                # Add flipped image
                img_flip = cv2.flip(img, 1)
                imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))
            if imgs:
                self.monsters[monster_name] = imgs
            else:
                logger.error(f"No images found in monster/{monster_name}/{monster_name}*")
                raise RuntimeError(f"No images found in monster/{monster_name}/{monster_name}*")
        logger.info(f"Loaded monsters: {list(self.monsters.keys())}")

        # Start keyboard controller thread
        self.kb = KeyBoardController(self.cfg, args)
        if args.disable_control:
            self.kb.disable()

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

        # Start health monitoring thread
        self.health_monitor = HealthMonitor(self.cfg, args, self.kb)
        if self.cfg["health_monitor"]["enable"]:
            self.health_monitor.start()

        # PyPy 優化：預熱 JIT
        self._warmup_jit()

    def _warmup_jit(self):
        """預熱 PyPy JIT 編譯器"""
        logger.info("預熱 PyPy JIT 編譯器...")
        
        # 執行一些計算密集型操作來預熱 JIT
        for i in range(1000):
            # 模擬圖像處理操作
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.GaussianBlur(dummy_img, (5, 5), 0)
            cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
        
        logger.info("JIT 預熱完成")

    def get_player_location_by_nametag(self):
        '''
        PyPy 優化版本的玩家位置檢測
        '''
        # 使用預分配的緩衝區
        if self.img_frame_gray_buffer is None:
            self.img_frame_gray_buffer = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)
        else:
            cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY, dst=self.img_frame_gray_buffer)
        
        # Get camera region in the game window
        img_camera = self.img_frame_gray_buffer[
            self.cfg["camera"]["y_start"]:self.cfg["camera"]["y_end"], :]

        # Get nametag image and search image
        if self.cfg["nametag"]["mode"] == "white_mask":
            # Apply Gaussian blur for smoother white detection
            img_camera = cv2.GaussianBlur(img_camera, (3, 3), 0)
            img_nametag = cv2.GaussianBlur(self.img_nametag_gray, (3, 3), 0)
            lower_white, upper_white = (150, 255)
            img_roi = cv2.inRange(img_camera, lower_white, upper_white)
            img_nametag  = cv2.inRange(img_nametag, lower_white, upper_white)
        elif self.cfg["nametag"]["mode"] == "grayscale":
            img_roi = img_camera
            img_nametag = self.img_nametag_gray
        else:
            logger.error(f"Unsupported nametag detection mode: {self.cfg['nametag']['mode']}")
            return

        # Pad search region to deal with fail detection when player is at map edge
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Get last frame name tag location
        if self.is_first_frame:
            last_result = None
        else:
            last_result = (
                self.loc_nametag[0] + pad_x,
                self.loc_nametag[1] + pad_y - self.cfg["camera"]["y_start"]
            )

        # Get number of splits
        h, w = img_nametag.shape
        num_splits = max(1, w // self.cfg["nametag"]["split_width"])
        w_split = w // num_splits

        # Get nametag's background mask
        mask = get_mask(self.img_nametag, (0, 255, 0))

        # Vertically split the nametag image
        nametag_splits = {}
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else w
            nametag_splits[f"{i+1}/{num_splits}"] = {
                "img": img_nametag[:, x_s:x_e],
                "mask": mask[:, x_s:x_e],
                "last_result": (
                    (last_result[0] + x_s, last_result[1]) if last_result else None
                ),
                "score_penalty": 0.0,
                "offset_x": x_s
            }

        # Match tempalte
        matches = []
        for tag_type, split in nametag_splits.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi,
                split["img"],
                last_result=split["last_result"],
                mask=split["mask"],
                global_threshold=self.cfg["nametag"]["global_diff_thres"]
            )
            w_match = split["img"].shape[1]
            h_match = split["img"].shape[0]
            score += split["score_penalty"]
            matches.append((tag_type, loc, score, w_match, h_match, is_cached, split["offset_x"]))

        # Select best match and fix offset:
        matches.sort(key=lambda x: (not x[5], x[2]))  # prefer cached, then low score
        tag_type, loc_nametag, score, w_match, h_match, is_cached, offset_x = matches[0]

        # Adjust match location back to full nametag coordinates
        loc_nametag = (loc_nametag[0] - offset_x, loc_nametag[1])
        loc_nametag = (
            loc_nametag[0] - pad_x,
            loc_nametag[1] - pad_y + self.cfg["camera"]["y_start"]
        )

        # Only update nametag location when score is good enough
        if score < self.cfg["nametag"]["diff_thres"]:
            self.loc_nametag = loc_nametag

        loc_player = (
            self.loc_nametag[0] - self.cfg["nametag"]["offset"][0],
            self.loc_nametag[1] - self.cfg["nametag"]["offset"][1]
        )

        # Draw name tag detection box for debugging
        draw_rectangle(self.img_frame_debug, self.loc_nametag,
                       self.img_nametag.shape, (0, 255, 0), "")
        text = f"NameTag,{round(score, 2)}," + \
                f"{'cached' if is_cached else 'missed'}," + \
                f"{tag_type}"
        cv2.putText(self.img_frame_debug, text,
                    (self.loc_nametag[0],
                     self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw player center for debugging
        cv2.circle(self.img_frame_debug,
                loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        return loc_player

    def switch_status(self, new_status):
        '''
        Switch to new status and log the transition.
        '''
        # Ignore dummy transition
        if self.status == new_status:
            return

        t_elapsed = round(time.time() - self.t_last_switch_status)
        logger.info(f"[switch_status] From {self.status}({t_elapsed} sec) to {new_status}.")
        self.status = new_status
        self.t_last_switch_status = time.time()

    def run_once(self):
        '''
        PyPy 優化版本的單幀處理
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # Make sure resolution is as expected
        if self.cfg["game_window"]["size"] != self.frame.shape[:2]:
            text = f"Unexpeted window size: {self.frame.shape[:2]} (expect {self.cfg['game_window']['size']})"
            logger.error(text)
            return

        # Resize raw frame to (1296, 759) - 使用預分配緩衝區
        if self.img_frame_buffer is None:
            self.img_frame_buffer = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)
        else:
            cv2.resize(self.frame, (1296, 759), dst=self.img_frame_buffer, interpolation=cv2.INTER_NEAREST)
        
        self.img_frame = self.img_frame_buffer

        # Get minimap coordinate and size on game window
        minimap_result = get_minimap_loc_size(self.img_frame)
        if minimap_result is None:
            logger.warning("Failed to get minimap location and size.")
        else:
            x, y, w, h = minimap_result
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]

        # Grayscale game window - 使用預分配緩衝區
        if self.img_frame_gray_buffer is None:
            self.img_frame_gray_buffer = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)
        else:
            cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY, dst=self.img_frame_gray_buffer)
        
        self.img_frame_gray = self.img_frame_gray_buffer

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if not self.args.patrol:
            self.img_route = self.img_routes[self.idx_routes]
            self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["camera"]["y_end"]:, :])

        # Draw HP/MP/EXP bar on debug window
        ratio_bars = [self.health_monitor.hp_ratio,
                      self.health_monitor.mp_ratio,
                      self.health_monitor.exp_ratio]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            x_s, y_s = (250, 90)
            # Print bar ratio on debug window
            cv2.putText(self.img_frame_debug,
                        f"{bar_name}: {ratio_bars[i]*100:.1f}%",
                        (x_s, y_s + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw bar on debug window
            x_s, y_s = (410, 73)
            # print(self.health_monitor.loc_size_bars)
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s+30*i:y_s+h+30*i, x_s:x_s+w] = \
                self.img_frame[self.cfg["camera"]["y_end"]:, :][y:y+h, x:x+w]

        # Check whether "Please remove runes" warning appears on screen
        if self.is_rune_warning():
            self.rune_detect_level = 0
            self.switch_status("finding_rune") # Stop hunting and start find runes

        # Get player location in game window
        self.loc_player = self.get_player_location_by_nametag()

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(
                                self.img_minimap,
                                minimap_player_color=self.cfg["minimap"]["player_color"])
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Get other player location on minimap & change channel
        loc_other_players = get_all_other_player_locations_on_minimap(self.img_minimap)
        if loc_other_players:
            # Calculate center value
            xs = [x for (x, y) in loc_other_players]
            ys = [y for (x, y) in loc_other_players]
            if len(xs) == 0 or len(ys) == 0:
                return 
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            if np.isnan(center_x) or np.isnan(center_y):
                return
            center = (int(np.mean(xs)), int(np.mean(ys)))
            
            # Change channel
            if self.cfg["auto_change_channel"] == "true":
                logger.warning("Player detected, immediately change channel.")
                self.kb.set_command("stop")
                self.kb.disable()
                time.sleep(1)
                self.channel_change()
                self.red_dot_center_prev = None
                return
            elif self.cfg["auto_change_channel"] == "pixel":
                if self.red_dot_center_prev is not None:
                    dx = abs(center[0] - self.red_dot_center_prev[0])
                    dy = abs(center[1] - self.red_dot_center_prev[1])
                    total = dx + dy
                    logger.debug(f"[RedDot] Movement dx={dx}, dy={dy}, total={total}")
                    if total > self.cfg["other_player_move_pixel"]:
                        logger.warning(f"Other player movement > {self.cfg['other_player_move_pixel']}px detected, triggering channel change.")
                        self.kb.set_command("stop")
                        self.kb.disable()
                        time.sleep(1)
                        self.channel_change() 
                        self.red_dot_center_prev = None  
                        return
                else:
                    self.red_dot_center_prev = center
        else:
            self.red_dot_center_prev = None  

        # Get player location on global map
        if self.args.patrol:
            self.loc_player_global = self.loc_player_minimap
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        # Check whether a rune icon is near player
        if self.status == "finding_rune" and self.is_rune_near_player():
            self.switch_status("near_rune")

        # Check whether we entered the rune mini-game
        if self.status == "near_rune" and (not self.args.disable_control) and \
            time.time() - self.t_last_rune_trigger > self.cfg["rune_find"]["rune_trigger_cooldown"]:
            self.kb.set_command("stop") # stop character
            time.sleep(0.1) # Wait for character to stop
            self.kb.disable() # Disable kb thread during rune solving

            # Attempt to trigger rune
            self.kb.press_key("up", 0.02)
            time.sleep(1) # Wait for rune game to pop up

            # If entered the game, start solving rune
            print("🔍 Checking if in rune game: ", self.is_in_rune_game())
            if self.is_in_rune_game():
                print("🔍 Entered rune game")
                self.solve_rune() # Blocking until runes solved
                self.rune_detect_level = 0 # reset rune detect level
                self.switch_status("hunting")

            # Restore kb thread
            self.kb.enable()

            self.t_last_rune_trigger = time.time()

        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.args.attack == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
        elif self.args.attack == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
        else:
            logger.error(f"Unsupported attack mode: {self.args.attack}")
            return
        x0 = max(0, self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0, self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        if self.status == "hunting":
            self.monster_info = self.get_monsters_in_range((x0, y0), (x1, y1))
        else:
            self.monster_info = []

        # Get attack direction
        if self.args.attack == "aoe_skill":
            if len(self.monster_info) == 0:
                attack_direction = None
            else:
                attack_direction = "I don't care"
            nearest_monster = self.get_nearest_monster()

        elif self.args.attack == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Compute distance for left
            distance_left = float('inf')
            if monster_left is not None:
                mx, my = monster_left["position"]
                mw, mh = monster_left["size"]
                center_left = (mx + mw // 2, my + mh // 2)
                distance_left = abs(center_left[0] - self.loc_player[0]) + \
                                abs(center_left[1] - self.loc_player[1])
            # Compute distance for right
            distance_right = float('inf')
            if monster_right is not None:
                mx, my = monster_right["position"]
                mw, mh = monster_right["size"]
                center_right = (mx + mw // 2, my + mh // 2)
                distance_right = abs(center_right[0] - self.loc_player[0]) + \
                                abs(center_right[1] - self.loc_player[1])
            # Choose attack direction and nearest monster
            attack_direction = None
            nearest_monster = None
            
            # Additional validation: check if monster is actually on the correct side
            def is_monster_on_correct_side(monster, direction):
                if monster is None:
                    return False
                mx, my = monster["position"]
                mw, mh = monster["size"]
                monster_center_x = mx + mw // 2
                player_x = self.loc_player[0]
                
                if direction == "left":
                    return monster_center_x < player_x  # Monster should be left of player
                else:  # direction == "right"
                    return monster_center_x > player_x  # Monster should be right of player
            
            # Only choose direction if there's a clear winner and monster is on correct side
            if monster_left is not None and monster_right is None and is_monster_on_correct_side(monster_left, "left"):
                attack_direction = "left"
                nearest_monster = monster_left
            elif monster_right is not None and monster_left is None and is_monster_on_correct_side(monster_right, "right"):
                attack_direction = "right"
                nearest_monster = monster_right
            elif monster_left is not None and monster_right is not None:
                # Both sides have monsters, check distance and side validation
                left_valid = is_monster_on_correct_side(monster_left, "left")
                right_valid = is_monster_on_correct_side(monster_right, "right")
                
                if left_valid and not right_valid:
                    attack_direction = "left"
                    nearest_monster = monster_left
                elif right_valid and not left_valid:
                    attack_direction = "right"
                    nearest_monster = monster_right
                elif left_valid and right_valid and distance_left < distance_right - 50:
                    attack_direction = "left"
                    nearest_monster = monster_left
                elif left_valid and right_valid and distance_right < distance_left - 50:
                    attack_direction = "right"
                    nearest_monster = monster_right
                # If both valid but distances too close, don't attack to avoid confusion
            
            # Debug attack direction selection
            if monster_left is not None or monster_right is not None:
                left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
                right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
                debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
                cv2.putText(self.img_frame_debug, debug_text,
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        command = ""
        if self.args.patrol:
            x, y = self.loc_player
            h, w = self.img_frame.shape[:2]
            loc_player_ratio = float(x)/float(w)
            left_ratio, right_ratio = self.cfg["patrol"]["range"]

            # Check if we need to change patrol direction
            if self.is_patrol_to_left and loc_player_ratio < left_ratio:
                self.patrol_turn_point_cnt += 1
            elif (not self.is_patrol_to_left) and loc_player_ratio > right_ratio:
                self.patrol_turn_point_cnt += 1

            if self.patrol_turn_point_cnt > self.cfg["patrol"]["turn_point_thres"]:
                self.is_patrol_to_left = not self.is_patrol_to_left
                self.patrol_turn_point_cnt = 0

            # Set command for patrol mode
            # Use proper attack range checking instead of just checking if monsters exist
            if (time.time() - self.t_patrol_last_attack > self.cfg["patrol"]["patrol_attack_interval"] and 
                len(self.monster_info) > 0 and nearest_monster is not None):
                # Check if monster is actually in attack range
                if attack_direction == "I don't care" or attack_direction == "left" or attack_direction == "right":
                    command = "attack"
                    self.t_patrol_last_attack = time.time()
            elif self.is_patrol_to_left:
                command = "walk left"
            else:
                command = "walk right"

        else:
            # get color code from img_route
            color_code = self.get_nearest_color_code()
            if color_code:
                if color_code["action"] == "goal":
                    # Switch to next route map
                    self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
                    logger.debug(f"Change to new route:{self.idx_routes}")
                command = color_code["action"]

            # teleport away from edge to avoid falling off cliff
            if self.is_near_edge() and \
                time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
                command = command.replace("walk", "teleport")
                self.t_last_teleport = time.time() # update timer

        if self.cfg["key"]["teleport"] == "": # disable teleport skill
            command = command.replace("teleport", "jump")

        # Special logic for each status, overwrite color code action
        if self.status == "hunting":
            # Perform a random action when player stuck
            if not self.args.patrol and self.is_player_stuck():
                command = self.get_random_action()
            elif command in ["up", "down", "jump right", "jump left"]:
                pass # Don't attack or heal while character is on rope or jumping
            # Note: HP/MP monitoring is now handled by separate HealthMonitor thread
            elif attack_direction == "I don't care" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack"
                self.t_last_attack = time.time()
            elif attack_direction == "left" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack left"
                self.t_last_attack = time.time()
            elif attack_direction == "right" and nearest_monster is not None and \
                time.time() - self.t_last_attack > self.cfg["directional_attack"]["cooldown"]:
                command = "attack right"
                self.t_last_attack = time.time()

        elif self.status == "finding_rune":
            if self.is_player_stuck():
                command = self.get_random_action()

            # If the HP is reduced switch to hurting (other player probably help solved the rune)
            if time.time() - self.health_monitor.last_hp_reduce_time < 3:
                self.switch_status("hunting")

            # Check if finding rune timeout
            if time.time() - self.t_last_switch_status > self.cfg["rune_find"]["timeout"]:
                self.switch_status("resting")

        elif self.status == "near_rune":
            # Stay in near_rune status for only a few seconds
            if time.time() - self.t_last_switch_status > self.cfg["rune_find"]["near_rune_duration"]:
                self.switch_status("hunting")

        elif self.status == "resting":
            self.img_routes = [self.img_route_rest] # Set up resting route
            self.idx_routes = 0

        else:
            logger.error(f"Unknown status: {self.status}")

        # send command to keyboard controller
        self.kb.set_command(command)

        # Debug: show current command on screen
        if command and len(command) > 0:
            cv2.putText(self.img_frame_debug, f"CMD: {command}",
                       (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Check if need to save screenshot
        if self.kb.is_need_screen_shot:
            screenshot(self.img_frame)
            self.kb.is_need_screen_shot = False

        # Enable cached location since second frame
        self.is_first_frame = False

        # 定期記錄性能資訊
        if time.time() % 60 < 0.1:  # 每分鐘記錄一次
            self.performance_monitor.log_performance()

        #####################
        ### Debug Windows ###
        #####################
        # Don't show debug window to save system resource
        if not self.cfg["system"]["show_debug_window"]:
            return

        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Resize img_route_debug for better visualization
        if not self.args.patrol:
            self.img_route_debug = cv2.resize(
                        self.img_route_debug, (0, 0),
                        fx=self.cfg["minimap"]["debug_window_upscale"],
                        fy=self.cfg["minimap"]["debug_window_upscale"],
                        interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Route Map Debug", self.img_route_debug)

    # 其他方法保持不變，但需要從原始檔案複製過來
    def get_player_location_on_global_map(self):
        '''從原始檔案複製'''
        pass
    
    def get_nearest_color_code(self):
        '''從原始檔案複製'''
        pass
    
    def get_nearest_monster(self, is_left = True, overlap_threshold=0.5):
        '''從原始檔案複製'''
        pass
    
    def get_monsters_in_range(self, top_left, bottom_right):
        '''從原始檔案複製'''
        pass
    
    def solve_rune(self):
        '''從原始檔案複製'''
        pass
    
    def is_player_stuck(self):
        '''從原始檔案複製'''
        pass
    
    def is_rune_warning(self):
        '''從原始檔案複製'''
        pass
    
    def is_rune_near_player(self):
        '''從原始檔案複製'''
        pass
    
    def is_in_rune_game(self):
        '''從原始檔案複製'''
        pass
    
    def is_near_edge(self):
        '''從原始檔案複製'''
        pass
    
    def get_random_action(self):
        '''從原始檔案複製'''
        pass
    
    def update_info_on_img_frame_debug(self):
        '''從原始檔案複製'''
        pass
    
    def update_img_frame_debug(self):
        '''從原始檔案複製'''
        pass
    
    def click_in_game_window(self, x, y):
        '''從原始檔案複製'''
        pass
        
    def channel_change(self):
        '''從原始檔案複製'''
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--patrol',
        action='store_true',
        help='Enable patrol mode'
    )

    # Argument to specify map name
    parser.add_argument(
        '--map',
        type=str,
        default='lost_time_1',
        help='Specify the map name'
    )

    parser.add_argument(
        "--monsters",
        type=str,
        default="evolved_ghost",
        help="Specify which monsters to load, comma-separated"
             "(e.g., --monsters green_mushroom,zombie_mushroom)"
    )

    parser.add_argument(
        '--attack',
        type=str,
        default='directional',
        help='Choose attack method, "directional", "aoe_skill"'
    )

    parser.add_argument(
        '--nametag',
        type=str,
        default='example',
        help='Choose nametag png file in nametag/'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='edit_me',
        help='Choose customized config yaml file in config/'
    )

    try:
        mapleStoryBot = MapleStoryBotPyPy(parser.parse_args())
    except Exception as e:
        logger.error(f"MapleStoryBotPyPy Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            mapleStoryBot.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / mapleStoryBot.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows() 