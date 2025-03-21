ROOT_DIR="src/"

echo -e "\n\nRunning ISORT to sort imports ..."
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=120 $ROOT_DIR

echo -e "\n\nRunning BLACK to format code ..."
black --line-length 120 $ROOT_DIR

echo -e "\n\nRunning autopep8 check ..."
autopep8 --in-place --aggressive --recursive $ROOT_DIR

echo -e "\n\nRunning PYLINT check ..."
pylint --disable duplicate-code,invalid-name,missing-module-docstring,missing-class-docstring,missing-function-docstring,too-many-branches,too-many-arguments,too-many-locals,too-many-statements,too-many-instance-attributes,too-few-public-methods,too-many-public-methods,too-many-lines $ROOT_DIR
