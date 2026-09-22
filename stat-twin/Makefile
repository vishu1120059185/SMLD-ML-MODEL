.PHONY: setup data features train eval report app test reproduce \
       e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 clean

PROFILE ?= fast
DS ?= FD001
MODEL ?= xgb
PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]" --quiet
	$(PYTHON) -m ruff check src/ tests/ --fix

data:
	$(PYTHON) -m stattwin.cli data --ds $(DS) --profile $(PROFILE)

features:
	$(PYTHON) -m stattwin.cli features --ds $(DS) --profile $(PROFILE)

train:
	$(PYTHON) -m stattwin.cli train --ds $(DS) --model $(MODEL) --profile $(PROFILE)

eval:
	$(PYTHON) -m stattwin.cli eval --ds $(DS) --profile $(PROFILE)

e0:
	$(PYTHON) -m stattwin.experiments.e0_data_audit --ds $(DS) --profile $(PROFILE)

e1:
	$(PYTHON) -m stattwin.experiments.e1_health_index --ds $(DS) --profile $(PROFILE)

e2:
	$(PYTHON) -m stattwin.experiments.e2_model_comparison --ds $(DS) --profile $(PROFILE)

e3:
	$(PYTHON) -m stattwin.experiments.e3_early_warning --ds $(DS) --profile $(PROFILE)

e4:
	$(PYTHON) -m stattwin.experiments.e4_ablation --ds $(DS) --profile $(PROFILE)

e5:
	$(PYTHON) -m stattwin.experiments.e5_uncertainty --ds $(DS) --profile $(PROFILE)

e6:
	$(PYTHON) -m stattwin.experiments.e6_generalization --profile $(PROFILE)

e7:
	$(PYTHON) -m stattwin.experiments.e7_operating_conditions --profile $(PROFILE)

e8:
	$(PYTHON) -m stattwin.experiments.e8_fault_injection --ds $(DS) --profile $(PROFILE)

e9:
	$(PYTHON) -m stattwin.experiments.e9_uncertainty_comparison --ds $(DS) --profile $(PROFILE)

app-data:
	$(PYTHON) -m stattwin.cli app-data --ds $(DS) --profile $(PROFILE)

app:
	$(PYTHON) -m streamlit run src/stattwin/dashboard/app.py --server.themeBase dark

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-leakage:
	$(PYTHON) -m pytest tests/leakage_guard/ -v --tb=short

report:
	$(PYTHON) -m stattwin.reports.generate --ds $(DS)

reproduce:
	$(PYTHON) -m stattwin.cli reproduce --profile full

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, True) for p in ['artifacts','results','reports/generated']]"
