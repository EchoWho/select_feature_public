#!/usr/bin/bash 
CURRENT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="$1:${PATH:+"$PATH"}"
    fi
}
pypathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        PYTHONPATH="$1:${PYTHONPATH:+":$PYTHONPATH"}"
    fi
}

export SFP_ROOT=$CURRENT_DIR
pypathadd $SFP_ROOT/

