#!/usr/bin/env bash
set -euo pipefail

GYMUSER_NAME="gymuser"
GYMUSER_HOME="/home/${GYMUSER_NAME}"

pick_reference_path() {
  local candidate
  for candidate in \
    "${GYMUSER_HOME}/rl_lib" \
    "${GYMUSER_HOME}/extreme-parkour" \
    "${GYMUSER_HOME}/robot_firmware"
  do
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

sync_gymuser_ids() {
  local reference_path target_uid target_gid current_uid current_gid existing_group changed

  if ! reference_path="$(pick_reference_path)"; then
    return 0
  fi

  target_uid="$(stat -c '%u' "${reference_path}")"
  target_gid="$(stat -c '%g' "${reference_path}")"
  current_uid="$(id -u "${GYMUSER_NAME}")"
  current_gid="$(id -g "${GYMUSER_NAME}")"
  changed=0

  if [[ "${target_gid}" != "${current_gid}" ]]; then
    if existing_group="$(getent group "${target_gid}" | cut -d: -f1)"; then
      usermod -g "${target_gid}" "${GYMUSER_NAME}"
    else
      groupmod -g "${target_gid}" "${GYMUSER_NAME}"
    fi
    changed=1
  fi

  if [[ "${target_uid}" != "${current_uid}" ]]; then
    usermod -o -u "${target_uid}" "${GYMUSER_NAME}"
    changed=1
  fi

  if [[ "${changed}" == "1" ]]; then
    find "${GYMUSER_HOME}" \
      \( -path "${GYMUSER_HOME}/rl_lib" -o -path "${GYMUSER_HOME}/extreme-parkour" -o -path "${GYMUSER_HOME}/robot_firmware" \) -prune \
      -o -exec chown -h "${target_uid}:${target_gid}" {} +
  fi
}

sync_gymuser_ids

export HOME="${GYMUSER_HOME}"
export USER="${GYMUSER_NAME}"
export PATH="/opt/conda/bin:${PATH}"

exec sudo -E -H -u "${GYMUSER_NAME}" /usr/bin/env \
  HOME="${GYMUSER_HOME}" \
  USER="${GYMUSER_NAME}" \
  PATH="${PATH}" \
  "$@"
