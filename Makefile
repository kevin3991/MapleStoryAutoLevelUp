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
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map fire_land_1 --monsters fire_pig,black_axe_stump --attack directional
run-2:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear --attack directional
run-3:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost --attack directional
run-4:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map north_forest_training_ground_8 --monsters wind_single_eye_beast --attack directional

test-solve-rune:
	$(ACTIVATE); $(PYTHON) test_solve_rune.py
