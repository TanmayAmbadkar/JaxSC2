"""Versioned reward weights, retained in run metadata for reproducibility."""

REWARD_PROFILES = {
    "legacy": {
        "navigation_progress": 2.0, "combat_progress": 0.5,
        "beacon_win": 25.0, "combat_win": 10.0,
        "enemy_damage": 0.005, "ally_damage": 0.005, "enemy_kill": 0.2,
        "defeat": -10.0, "timeout": -15.0,
    },
    "balanced": {
        "navigation_progress": 2.0, "combat_progress": 2.0,
        "beacon_win": 25.0, "combat_win": 25.0,
        "enemy_damage": 0.005, "ally_damage": 0.005, "enemy_kill": 0.2,
        "defeat": -10.0, "timeout": -15.0,
    },
}
