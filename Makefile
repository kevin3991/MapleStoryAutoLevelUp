PYTHON=python3
PYPY=pypy3
VENV=venv
VENV_PYPY=venv_pypy
ACTIVATE=. $(VENV)/bin/activate
ACTIVATE_PYPY=. $(VENV_PYPY)/bin/activate

.PHONY: setup clean run test-solve-rune setup-pypy run-pypy clean-pypy

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE); pip install --upgrade pip
	$(ACTIVATE); pip install -r requirements.txt

setup-pypy:
	$(PYPY) -m venv $(VENV_PYPY)
	$(ACTIVATE_PYPY); pip install --upgrade pip
	$(ACTIVATE_PYPY); pip install -r requirements.txt

clean:
	rm -rf $(VENV)

clean-pypy:
	rm -rf $(VENV_PYPY)

run:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack aoe_skill

run-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack aoe_skill

run-empty-house:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map empty_house --monsters mushroom --attack directional

run-empty-house-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map empty_house --monsters mushroom --attack directional

run-cloud-balcony:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear --attack directional

run-cloud-balcony-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear --attack directional

run-north-forest-training-ground-2:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map north_forest_training_ground_2 --monsters green_mushroom,spike_mushroom --attack directional

run-north-forest-training-ground-2-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map north_forest_training_ground_2 --monsters green_mushroom,spike_mushroom --attack directional

run-3:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost --attack directional

run-3-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost --attack directional

run-4:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map north_forest_training_ground_8 --monsters wind_single_eye_beast --attack directional

run-4-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map north_forest_training_ground_8 --monsters wind_single_eye_beast --attack directional

run-5:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map monkey_swamp_3 --monsters angel_monkey --attack aoe_skill

run-5-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map monkey_swamp_3 --monsters angel_monkey --attack aoe_skill

run-6:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map ant_cave_2 --monsters spike_mushroom,zombie_mushroom --attack aoe_skill

run-6-pypy:
	$(ACTIVATE_PYPY); $(PYPY) mapleStoryAutoLevelUp.py --map ant_cave_2 --monsters spike_mushroom,zombie_mushroom --attack aoe_skill

test-solve-rune:
	$(ACTIVATE); $(PYTHON) test_solve_rune.py

test-solve-rune-pypy:
	$(ACTIVATE_PYPY); $(PYPY) test_solve_rune.py

# 性能測試
benchmark:
	$(ACTIVATE); $(PYTHON) test_performance.py

benchmark-pypy:
	$(ACTIVATE_PYPY); $(PYPY) test_performance.py

# 快速啟動 PyPy 版本
quick-pypy:
	$(PYPY) mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig --attack aoe_skill
