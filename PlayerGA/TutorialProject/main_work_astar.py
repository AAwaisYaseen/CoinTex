import kivy.app
import kivy.uix.screenmanager
import kivy.uix.image
import csv
import heapq
import math
import random
import kivy.core.audio
import os
import functools
import kivy.uix.behaviors
import pickle
import threading
import kivy.base
import time



ALGORITHM_NAME = "hazard_aware_astar_optimized"
LEVEL_TIMEOUT_SECONDS = 30.0
GRID_SIZE = 21
MAX_MOVE = 8
MONSTER_RANGE = 0.20
FIRE_RANGE = 0.12
MONSTER_COST = 18.0
FIRE_COST = 20.0


class CollectCoinThread(threading.Thread):


    def __init__(self, screen):
        super().__init__(daemon=True)
        self.screen = screen


    @staticmethod
    def to_grid(position):
        last_cell = GRID_SIZE - 1
        x = min(max(round(position[0] * last_cell), 0), last_cell)
        y = min(max(round(position[1] * last_cell), 0), last_cell)
        return x, y

    @staticmethod
    def to_position(cell):
        last_cell = GRID_SIZE - 1
        return [cell[0] / last_cell, cell[1] / last_cell]

    @staticmethod
    def estimate(cell, goals):
        return min(math.dist(cell, goal) for goal in goals)

    @staticmethod
    def danger_cost(cell, monsters, fires):
        position = CollectCoinThread.to_position(cell)
        cost = 0

        for monster in monsters:
            distance = math.dist(position, monster)
            if distance < MONSTER_RANGE:
                cost += MONSTER_COST * (1 + (MONSTER_RANGE - distance) / MONSTER_RANGE)
            elif distance < MONSTER_RANGE * 2:
                cost += MONSTER_COST * 0.25 * (
                    (MONSTER_RANGE * 2 - distance) / MONSTER_RANGE
                )

        for fire in fires:
            distance = math.dist(position, fire)
            if distance < FIRE_RANGE:
                cost += FIRE_COST * (1 + (FIRE_RANGE - distance) / FIRE_RANGE)
            elif distance < FIRE_RANGE * 2:
                cost += FIRE_COST * 0.25 * (
                    (FIRE_RANGE * 2 - distance) / FIRE_RANGE
                )

        return cost

    @staticmethod
    def get_moves(cell):
        for dx, dy in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),             (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ):
            next_cell = cell[0] + dx, cell[1] + dy
            if 0 <= next_cell[0] < GRID_SIZE and 0 <= next_cell[1] < GRID_SIZE:
                move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1
                yield next_cell, move_cost

    @staticmethod
    def build_path(previous, current):
        path = [current]
        while current in previous:
            current = previous[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def astar(start_pos, coins, monsters, fires):
        start = CollectCoinThread.to_grid(start_pos)
        goals = {CollectCoinThread.to_grid(coin) for coin in coins}
        open_list = []
        heapq.heappush(open_list, (CollectCoinThread.estimate(start, goals), 0, start))

        previous = {}
        costs = {start: 0}

        while open_list:
            _, old_cost, current = heapq.heappop(open_list)
            if old_cost > costs[current]:
                continue

            if current in goals:
                coin_position = min(
                    coins,
                    key=lambda coin: math.dist(
                        CollectCoinThread.to_position(current), coin
                    ),
                )
                return CollectCoinThread.build_path(previous, current), coin_position

            for next_cell, move_cost in CollectCoinThread.get_moves(current):
                new_cost = costs[current] + move_cost
                new_cost += CollectCoinThread.danger_cost(next_cell, monsters, fires)

                if new_cost >= costs.get(next_cell, float("inf")):
                    continue

                previous[next_cell] = current
                costs[next_cell] = new_cost
                total = new_cost + CollectCoinThread.estimate(next_cell, goals)
                heapq.heappush(open_list, (total, new_cost, next_cell))

        return None, None

    @staticmethod
    def line_is_safe(start, end, monsters, fires):
        distance = math.dist(start, end)
        checks = max(2, round(distance * 40))

        for i in range(1, checks + 1):
            amount = i / checks
            point = [
                start[0] + (end[0] - start[0]) * amount,
                start[1] + (end[1] - start[1]) * amount,
            ]

            for monster in monsters:
                if math.dist(point, monster) < MONSTER_RANGE:
                    return False
            for fire in fires:
                if math.dist(point, fire) < FIRE_RANGE:
                    return False

        return True

    @staticmethod
    def next_point(path, coin, character, monsters, fires):
        if CollectCoinThread.line_is_safe(character, coin, monsters, fires):
            return coin

        if len(path) < 2:
            return coin

        last_step = min(MAX_MOVE, len(path) - 1)
        for step in range(last_step, 0, -1):
            target = CollectCoinThread.to_position(path[step])
            if step == len(path) - 1:
                target = coin
            if CollectCoinThread.line_is_safe(character, target, monsters, fires):
                return target

        return CollectCoinThread.to_position(path[1])



    def run(self):
        start_pos = [0.1, 0.1]
        app.start_char_animation(lvl_num, start_pos)
        time.sleep(self.screen.char_anim_duration + 0.05)

        while True:
            if self.screen.character_killed or self.screen.level_finished:
                return

            elapsed_time = time.perf_counter() - self.screen.level_start_time
            if elapsed_time >= LEVEL_TIMEOUT_SECONDS:
                self.screen.character_killed = True
                self.screen.level_finished = True
                app.save_result(lvl_num, "FAIL", "timeout")
                kivy.clock.Clock.schedule_once(
                    functools.partial(app.stop_timed_out_level, lvl_num), 0
                )
                return

            character, coins, monsters, fires = get_positions()
            if not coins:
                return

            path, coin = CollectCoinThread.astar(character, coins, monsters, fires)
            if not path:
                time.sleep(0.1)
                continue

            target = CollectCoinThread.next_point(
                path, coin, character, monsters, fires
            )
            app.start_char_animation(lvl_num, target)
            time.sleep(self.screen.char_anim_duration + 0.05)

            character = self.screen.ids['character_image_lvl'+str(lvl_num)]
            kivy.clock.Clock.schedule_once(
                lambda _dt, image=character: app.char_pos_hint(image), 0
            )
            time.sleep(0.02)


def get_center(widget):
    return [
        widget.pos_hint['x'] + widget.size_hint[0] / 2,
        widget.pos_hint['y'] + widget.size_hint[1] / 2,
    ]


def get_positions():
    curr_screen = app.root.screens[lvl_num]
    character = curr_screen.ids['character_image_lvl'+str(lvl_num)]
    coins = [get_center(coin) for coin in list(curr_screen.coins_ids.values())]

    monsters = []
    for i in range(curr_screen.num_monsters):
        monster = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(lvl_num)]
        monsters.append(get_center(monster))

    fires = []
    for i in range(curr_screen.num_fires):
        fire = curr_screen.ids['fire'+str(i+1)+'_lvl'+str(lvl_num)]
        fires.append(get_center(fire))

    return get_center(character), coins, monsters, fires







class CointexApp(kivy.app.App):

    def on_start(self):
        self.results_lock = threading.Lock()
        music_dir = os.getcwd()+"/music/"
        self.main_bg_music = kivy.core.audio.SoundLoader.load(music_dir+"bg_music_piano_flute.wav")
        self.main_bg_music.loop = True
        self.main_bg_music.volume = 0.1 #smaller the volume
        self.main_bg_music.play()

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def get_run_number(self, file_path, screen_num):
        run_number = 1
        if not os.path.exists(file_path):
            return run_number

        with open(file_path, newline="", encoding="utf-8") as result_file:
            for row in csv.DictReader(result_file):
                if (
                    row.get("algorithm") == ALGORITHM_NAME
                    and row.get("level") == str(screen_num)
                ):
                    run_number += 1
        return run_number

    def save_result(self, screen_num, result, reason=""):
        curr_screen = self.root.screens[screen_num]

        with self.results_lock:
            if curr_screen.result_recorded:
                return
            curr_screen.result_recorded = True
            curr_screen.level_finished = True

            elapsed_time = time.perf_counter() - curr_screen.level_start_time
            remaining_life = max(
                0,
                100 * (
                    1 - curr_screen.num_collisions_hit / curr_screen.num_collisions_level
                ),
            )

            result_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results"
            )
            os.makedirs(result_folder, exist_ok=True)
            file_name = "test_runs.csv"
            file_path = os.path.join(result_folder, file_name)
            columns = [
                "algorithm", "level", "run", "result", "coins_collected",
                "total_coins", "time_seconds", "total_hits", "monster_hits",
                "fire_hits", "remaining_life_percent", "failure_reason",
            ]

            run_number = self.get_run_number(file_path, screen_num)
            file_exists = os.path.exists(file_path)
            with open(file_path, "a", newline="", encoding="utf-8") as result_file:
                writer = csv.DictWriter(result_file, fieldnames=columns)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "algorithm": ALGORITHM_NAME,
                    "level": screen_num,
                    "run": run_number,
                    "result": result,
                    "coins_collected": curr_screen.num_coins_collected,
                    "total_coins": curr_screen.num_coins,
                    "time_seconds": f"{elapsed_time:.2f}",
                    "total_hits": curr_screen.num_collisions_hit,
                    "monster_hits": curr_screen.monster_hits,
                    "fire_hits": curr_screen.fire_hits,
                    "remaining_life_percent": f"{remaining_life:.1f}",
                    "failure_reason": reason,
                })

        shown_result = "FAIL (TIMEOUT)" if reason == "timeout" else result
        print("\n--- LEVEL RESULT ---")
        print(f"Algorithm: {ALGORITHM_NAME}")
        print(f"Result: {shown_result}")
        print(f"Coins: {curr_screen.num_coins_collected}/{curr_screen.num_coins}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Total hits: {curr_screen.num_collisions_hit}")
        print(f"Monster hits: {curr_screen.monster_hits}")
        print(f"Fire hits: {curr_screen.fire_hits}")
        print(f"Remaining life: {remaining_life:.1f}%")
        if reason:
            print(f"Failure reason: {reason}")
        print(f"Saved to: {file_name}")
        print("--------------------\n")

    def read_game_info(self):
        return 24, True

    def activate_levels(self, next_level_num, congrats_displayed_once):
        num_levels = len(self.root.screens[0].ids['lvls_imagebuttons'].children)

        levels_imagebuttons = self.root.screens[0].ids['lvls_imagebuttons'].children
        for i in range(num_levels-next_level_num, num_levels):
            levels_imagebuttons[i].disabled = False
            levels_imagebuttons[i].color = [1,1,1,1]

        for i in range(0, num_levels-next_level_num):
            levels_imagebuttons[i].disabled = True
            levels_imagebuttons[i].color = [1,1,1,0.5]

        if next_level_num == (num_levels+1) and congrats_displayed_once == False:
            self.root.current = "alllevelscompleted"

    def screen_on_pre_leave(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters): curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].pos_hint = {'x': 0.8, 'y': 0.8}
        curr_screen.ids['character_image_lvl'+str(screen_num)].pos_hint = {'x': 0.0, 'y': 0.0}

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def screen_on_pre_enter(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        curr_screen.level_start_time = time.perf_counter()
        curr_screen.character_killed = False
        curr_screen.level_finished = False
        curr_screen.result_recorded = False
        curr_screen.num_coins_collected = 0
        curr_screen.ids['character_image_lvl'+str(screen_num)].im_num = curr_screen.ids['character_image_lvl'+str(screen_num)].start_im_num
        for i in range(curr_screen.num_monsters): curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].im_num = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].start_im_num
        curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins 0/"+str(curr_screen.num_coins)
        curr_screen.ids['level_number_lvl'+str(screen_num)].text = "Level "+str(screen_num)

        curr_screen.num_collisions_hit = 0
        curr_screen.monster_hits = 0
        curr_screen.fire_hits = 0
        remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
        remaining_life_percent_lvl_widget.size_hint = (remaining_life_percent_lvl_widget.remaining_life_size_hint_x, remaining_life_percent_lvl_widget.size_hint[1])

        for i in range(curr_screen.num_fires): curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)].pos_hint = {'x': 1.1, 'y': 1.1}

        for key, coin in curr_screen.coins_ids.items():
            curr_screen.ids['layout_lvl'+str(screen_num)].remove_widget(coin)
        curr_screen.coins_ids = {}

        coin_width = 0.05
        coin_height = 0.05

        curr_screen = self.root.screens[screen_num]

        section_width = 1.0/curr_screen.num_coins
        for k in range(curr_screen.num_coins):
            x = random.uniform(section_width*k, section_width*(k+1)-coin_width)
            y = random.uniform(0, 1-coin_height)
            coin = kivy.uix.image.Image(source="other-images/coin.png", size_hint=(coin_width, coin_height), pos_hint={'x': x, 'y': y}, allow_stretch=True)
            curr_screen.ids['layout_lvl'+str(screen_num)].add_widget(coin, index=-1)
            curr_screen.coins_ids['coin'+str(k)] = coin

    def screen_on_enter(self, screen_num):
        music_dir = os.getcwd()+"/music/"
        self.bg_music = kivy.core.audio.SoundLoader.load(music_dir+"bg_music_piano.wav")
        # start to count the time when the agent is acting:
        time.perf_counter()
        self.bg_music.loop = True

        self.coin_sound = kivy.core.audio.SoundLoader.load(music_dir+"coin.wav")
        self.level_completed_sound = kivy.core.audio.SoundLoader.load(music_dir+"level_completed_flaute.wav")
        self.char_death_sound = kivy.core.audio.SoundLoader.load(music_dir+"char_death_flaute.wav")

        self.bg_music.volume = 0.1 #smaller the volume
        self.coin_sound.volume = 0.1 #smaller the volume
        self.level_completed_sound.volume = 0.1 #smaller the volume
        self.char_death_sound.volume = 0.1 #smaller the volume

        self.bg_music.play()

        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters):
            monster_image = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)]
            new_pos = (random.uniform(0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
            self.start_monst_animation(monster_image=monster_image, new_pos=new_pos, anim_duration=random.uniform(monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

        for i in range(curr_screen.num_fires):
            fire_widget = curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)]
            self.start_fire_animation(fire_widget=fire_widget, pos=(0.0, 0.5), anim_duration=5.0)

        global lvl_num

        lvl_num = screen_num

        collectCoinThread = CollectCoinThread(screen=curr_screen)
        collectCoinThread.start()

    def stop_timed_out_level(self, screen_num, *args):
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]

        self.bg_music.stop()
        kivy.animation.Animation.cancel_all(character_image)
        for i in range(curr_screen.num_monsters):
            kivy.animation.Animation.cancel_all(
                curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)]
            )
        for i in range(curr_screen.num_fires):
            kivy.animation.Animation.cancel_all(
                curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)]
            )

        kivy.clock.Clock.schedule_once(
            functools.partial(self.back_to_main_screen, curr_screen.parent), 3
        )

    def start_monst_animation(self, monster_image, new_pos, anim_duration):
        monst_anim = kivy.animation.Animation(pos_hint={'x': new_pos[0], 'y': new_pos[1]}, im_num=monster_image.end_im_num, duration=anim_duration)
        monst_anim.bind(on_complete=self.monst_animation_completed)
        monst_anim.start(monster_image)

    def monst_animation_completed(self, *args):
        monster_image = args[1]
        monster_image.im_num = monster_image.start_im_num

        new_pos = (random.uniform(0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
        self.start_monst_animation(monster_image=monster_image, new_pos= new_pos,anim_duration=random.uniform(monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

    def monst_pos_hint(self, monster_image):
        screen_num = int(monster_image.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]

        character_center = character_image.center
        monster_center = monster_image.center

        gab_x = character_image.width / 2
        gab_y = character_image.height / 2
        if character_image.collide_widget(monster_image) and abs(character_center[0] - monster_center[0]) <= gab_x and abs(character_center[1] - monster_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            curr_screen.monster_hits += 1
            life_percent = float(curr_screen.num_collisions_hit)/float(curr_screen.num_collisions_level)

#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"
            remaining_life_percent_lvl_widget=curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint =  (remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if (
                curr_screen.num_collisions_hit >= curr_screen.num_collisions_level
                and not curr_screen.level_finished
            ):
                self.save_result(screen_num, "FAIL", "collision_death")
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)

    def change_monst_im(self, monster_image):
        monster_image.source = "monsters-images/" + str(int(monster_image.im_num)) + ".png"

    def touch_down_handler(self, screen_num, args):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed == False:
            self.start_char_animation(screen_num, args[1].spos)

    def start_char_animation(self, screen_num, touch_pos):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed:
            return

        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]
        character_image.im_num = character_image.start_im_num
        char_anim = kivy.animation.Animation(pos_hint={'x': touch_pos[0] - character_image.size_hint[0] / 2,'y': touch_pos[1] - character_image.size_hint[1] / 2}, im_num=character_image.end_im_num, duration=curr_screen.char_anim_duration)
        char_anim.bind(on_complete=self.char_animation_completed)
        char_anim.start(character_image)

    def char_animation_completed(self, *args):
        character_image = args[1]
        character_image.im_num = character_image.start_im_num

    def char_pos_hint(self, character_image):
        screen_num = int(character_image.parent.parent.name[5:])
        character_center = character_image.center

        gab_x = character_image.width / 3
        gab_y = character_image.height / 3
        coins_to_delete = []
        curr_screen = self.root.screens[screen_num]

        for coin_key, curr_coin in curr_screen.coins_ids.items():
            curr_coin_center = curr_coin.center
            if character_image.collide_widget(curr_coin) and abs(character_center[0] - curr_coin_center[0]) <= gab_x and abs(character_center[1] - curr_coin_center[1]) <= gab_y:
                self.coin_sound.play()
                coins_to_delete.append(coin_key)
                curr_screen.ids['layout_lvl'+str(screen_num)].remove_widget(curr_coin)
                curr_screen.num_coins_collected = curr_screen.num_coins_collected + 1
                curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins "+str(curr_screen.num_coins_collected)+"/"+str(curr_screen.num_coins)
                if curr_screen.num_coins_collected == curr_screen.num_coins:
                    self.save_result(screen_num, "WIN")
                    self.bg_music.stop()
                    self.level_completed_sound.play()
                    kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)
                    for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                    for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                    next_level_num, congrats_displayed_once = self.read_game_info()
                    if (screen_num+1) > next_level_num:
                        game_info_file = open("game_info",'wb')
                        pickle.dump([{'lastlvl':screen_num+1, "congrats_displayed_once": False}], game_info_file)
                        game_info_file.close()
                    else:
                        game_info_file = open("game_info",'wb')
                        pickle.dump([{'lastlvl':next_level_num, "congrats_displayed_once": True}], game_info_file)
                        game_info_file.close()

        if len(coins_to_delete) > 0:
            for coin_key in coins_to_delete:
                del curr_screen.coins_ids[coin_key]

    def change_char_im(self, character_image):
        character_image.source = "character-images/" + str(int(character_image.im_num)) + ".png"

    def start_fire_animation(self, fire_widget, pos, anim_duration):
        fire_anim = kivy.animation.Animation(pos_hint=fire_widget.fire_start_pos_hint, duration=fire_widget.fire_anim_duration)+kivy.animation.Animation(pos_hint=fire_widget.fire_end_pos_hint, duration=fire_widget.fire_anim_duration)
        fire_anim.repeat = True
        fire_anim.start(fire_widget)

    def fire_pos_hint(self, fire_widget):
        screen_num = int(fire_widget.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]

        character_center = character_image.center
        fire_center = fire_widget.center

        gab_x = character_image.width / 3
        gab_y = character_image.height / 3
        if character_image.collide_widget(fire_widget) and abs(character_center[0] - fire_center[0]) <= gab_x and abs(character_center[1] - fire_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            curr_screen.fire_hits += 1
            life_percent = float(curr_screen.num_collisions_hit)/float(curr_screen.num_collisions_level)

            remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"

            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint =  (remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if (
                curr_screen.num_collisions_hit >= curr_screen.num_collisions_level
                and not curr_screen.level_finished
            ):
                self.save_result(screen_num, "FAIL", "collision_death")
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)

    def back_to_main_screen(self, screenManager, *args):
        screenManager.current = "main"

    def main_screen_on_enter(self):
        self.main_bg_music.play()

    def main_screen_on_leave(self):
        self.main_bg_music.stop()

class ImageButton(kivy.uix.behaviors.ButtonBehavior, kivy.uix.image.Image):
    pass

class MainScreen(kivy.uix.screenmanager.Screen):
    pass

class AboutUs(kivy.uix.screenmanager.Screen):
    pass

class AllLevelsCompleted(kivy.uix.screenmanager.Screen):
    pass

class Level1(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 5
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 20

class Level2(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 8
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level3(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level4(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 20

class Level5(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20

class Level6(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 20

class Level7(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level8(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level9(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level10(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 14
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level11(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30

class Level12(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30

class Level13(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20

class Level14(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30

class Level15(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 16
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 30

class Level16(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 35

class Level17(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level18(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level19(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30

class Level20(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 8
    num_collisions_hit = 0
    num_collisions_level = 30

class Level21(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 18
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level22(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level23(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 25
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30

class Level24(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30

app = CointexApp()
app.title = "CoinTex"
app.icon = 'cointex_logo.png'
# initialize the thread pool and create the runner

app.run()
