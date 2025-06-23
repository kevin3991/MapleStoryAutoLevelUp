PYTHON=python3
VENV=venv
ACTIVATE=. $(VENV)/bin/activate

.PHONY: setup clean run test-solve-rune

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE); pip install --upgrade pip
	$(ACTIVATE); pip install -r requirements.txt

clean:
	rm -rf $(VENV)

run:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack aoe_skill
run-empty-house:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map empty_house --monsters mushroom --attack directional
run-cloud-balcony:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear --attack directional
run-north-forest-training-ground-2:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map north_forest_training_ground_2 --monsters green_mushroom,spike_mushroom --attack directional
run-3:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost --attack directional
run-4:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map north_forest_training_ground_8 --monsters wind_single_eye_beast --attack directional
run-5:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map monkey_swamp_3 --monsters angel_monkey --attack aoe_skill
run-6:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map ant_cave_2 --monsters spike_mushroom,zombie_mushroom --attack aoe_skill

test-solve-rune:
	$(ACTIVATE); $(PYTHON) test_solve_rune.py
test-rune-detection:
	$(ACTIVATE); $(PYTHON) test_rune_detection.py
