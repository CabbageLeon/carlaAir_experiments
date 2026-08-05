#!/bin/bash
# CarlaAir launcher — interactive map picker when no map or unknown map given.

BINARY="./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
MAP_DIR="CarlaUE4/Content/Carla/Maps"
MAPS=$(find "$MAP_DIR" -name "Town*.umap" ! -name "*_Opt*" ! -name "*BuiltData*" \
       -exec basename {} .umap \; | sort)

resolve_map() {
    local arg="$1"
    # Already a full path
    [[ "$arg" == /Game/* ]] && { echo "$arg"; return 0; }
    # Try exact match
    for m in $MAPS; do
        [[ "$m" == "$arg" ]] && { echo "/Game/Carla/Maps/$m"; return 0; }
    done
    return 1
}

map=""
if [ -n "$1" ]; then
    map=$(resolve_map "$1")
    if [ -n "$map" ]; then
        shift
    fi
fi

if [ -z "$map" ]; then
    echo "可用地图:"
    i=1; declare -a opts
    while IFS= read -r name; do
        echo "  $i. $name"
        opts[$i]="$name"; ((i++))
    done <<< "$MAPS"
    default=6
    read -p "选择地图 [1-$((i-1)), 默认=$default]: " choice
    choice=${choice:-$default}
    selected=${opts[$choice]:-${opts[$default]}}
    map="/Game/Carla/Maps/$selected"
fi

echo "Starting $map ..."
exec "$BINARY" -windowed -ResX=1280 -ResY=720 "$map" "$@"
