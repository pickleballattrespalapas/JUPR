# JUPR UI Style Guide

## Theme
Use apply_clean_theme()

## Rendering Rules
- Escape HTML unless intentional
- Drop legacy story columns before render

## Admin Gating
Check ctx.admin_logged_in

## Performance
Do not run badge engine inside render()
