PYTHON=python3
VENV=venv
ACTIVATE=. $(VENV)/bin/activate

.PHONY: setup clean run

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE); pip install --upgrade pip
	$(ACTIVATE); pip install -r requirements.txt

clean:
	rm -rf $(VENV)

run:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp_legacy.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack magic_claw
run-2:
	$(ACTIVATE); $(PYTHON) mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear --attack magic_claw