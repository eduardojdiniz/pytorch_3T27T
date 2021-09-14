.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your bucket for syncing data (do not include 's3://')
PROFILE = default
PROJECT_NAME = pytorch_3T27T
PYTHON_INTERPRETER = python3
USERNAME = eduardojdiniz
EMAIL = edd32@pitt.edu
GITHUB_TOKEN_FILE?=
GITHUB_TOKEN?="$(cat "$(GITHUB_TOKEN_FILE)")"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806 `find . -name \*.py | grep -v setup.py | grep -v /docs/`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

## Upload Data to S3
sync_data_to_s3:
    ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
    else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
    endif

## Download Data from S3
sync_data_from_s3:
    ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
    else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
    endif

## Set up Git
git:
	@echo "starting git repo..."
	@if [ -d ".git" ]; then \
		echo ".git already exists"; \
	else \
		git init; \
		git config user.name $(USERNAME); \
		git config user.email $(EMAIL); \
		git add --all && git commit -m "Initial commit, hello world!"; \
		git branch -m main; \
    fi;

## Set up GitHub
github: git
	@echo "adding github remote..."
	@if [ -z $(GITHUB_TOKEN) ]; then \
		echo "Necessary to pass file with your personal GitHub access token as TOKEN_FILE=<path/to/file> or have the TOKEN environment variable with your token string set upon calling this target"; \
		echo ""; \
		echo "Please follow the instructions at https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token"; \
	else \
		curl -H "Authorization: token $(GITHUB_TOKEN)" https://api.github.com/user/repos \
			-d '{"name": "'$(PROJECT_NAME)'", "private": "false"}'; \
		git remote add origin git@github.com:$(USERNAME)/$(PROJECT_NAME).git; \
		echo "push main to origin"; \
		git push -u origin main; \
		echo "set origin as the default remote repo when on branch main"; \
		git config branch.main.remote origin; \
		echo "set origin main as the default remote branch when pull on the main branch"; \
		git config branch.main.merge refs/heads/main; \
		echo "push the current branch to a branch of the same name by default"; \
		git config push.default current; \
	fi;

## Set up python interpreter environment
virtual_environment:
    ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
    ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create -y -q --name $(PROJECT_NAME) python=3.8
    else
	conda create -y -q --name $(PROJECT_NAME) python=2.7
    endif
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
    else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
    endif

## Install Python Dependencies
install_requirements: test_environment
    ifeq (True,$(HAS_CONDA))
	conda env update -q --name $(PROJECT_NAME) --file environment.yml --prune
    else
	workon $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt
    endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# Test the entire pytorch_3T27T module
test:
	py.test --pyargs $(PROJECT_NAME) --cov-report term-missing --cov=$(PROJECT_NAME)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
