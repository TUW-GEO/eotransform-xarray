#!/bin/bash

EOTRANSFORM_TOKEN=$1
EOTRANSFORM_PANDAS_TOKEN=$2

if [ -f ~/.config/pip/pip.conf ]; then
	echo "pip.conf already exists, you've to add python indices manually."
else
	echo "creating pip.conf..."
	mkdir -p ~/.config/pip
	cat <<- EOF > ~/.config/pip/pip.conf
	[global]
	extra-index-url =
	  https://__token__:${EOTRANSFORM_TOKEN}@git.eodc.eu/api/v4/projects/717/packages/pypi/simple
	  https://__token__:${EOTRANSFORM_PANDAS_TOKEN}@git.eodc.eu/api/v4/projects/719/packages/pypi/simple
	EOF
fi