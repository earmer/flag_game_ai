/**
 * ctf_sim.h - CTF Game Simulator C Interface
 *
 * Pure C implementation for performance optimization.
 * Uses JSON strings for data exchange with Python.
 */

#ifndef CTF_SIM_H
#define CTF_SIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to simulator instance */
typedef void* CTFSimHandle;

/**
 * Create a new CTF simulator instance.
 *
 * @param width       Map width (default 20)
 * @param height      Map height (default 20)
 * @param num_players Players per team (default 3)
 * @param num_flags   Flags per team (default 9)
 * @param seed        Random seed (-1 for random)
 * @return Handle to simulator, or NULL on failure
 */
CTFSimHandle ctf_sim_create(int width, int height, int num_players, int num_flags, int seed);

/**
 * Destroy simulator and free all resources.
 */
void ctf_sim_destroy(CTFSimHandle sim);

/**
 * Reset the game to initial state.
 * Generates new map, obstacles, flags, and player positions.
 *
 * @return JSON string with init payload, or NULL on error
 *         Caller must NOT free this string (internal buffer)
 */
const char* ctf_sim_reset(CTFSimHandle sim);

/**
 * Execute one game step with given actions.
 *
 * @param actions_json JSON object: {"L0": "up", "L1": "right", ...}
 * @return JSON string with step result, or NULL on error
 */
const char* ctf_sim_step(CTFSimHandle sim, const char* actions_json);

/**
 * Execute one game step with binary actions (faster, no JSON parsing).
 *
 * @param actions Array of 6 action codes: [L0, L1, L2, R0, R1, R2]
 *                Action codes: 0=stay, 1=up, 2=down, 3=left, 4=right
 * @return "ok" on success, "done" if game finished, NULL on error
 */
const char* ctf_sim_step_binary(CTFSimHandle sim, const unsigned char* actions);

/**
 * Get current game status for a team.
 *
 * @param team "L" or "R"
 * @return JSON string with status, or NULL on error
 */
const char* ctf_sim_status(CTFSimHandle sim, const char* team);

/**
 * Get init payload for a team (map info, zones, etc.)
 *
 * @param team "L" or "R"
 * @return JSON string with init payload
 */
const char* ctf_sim_init_payload(CTFSimHandle sim, const char* team);

/**
 * Check if game is finished.
 * @return 1 if done, 0 otherwise
 */
int ctf_sim_done(CTFSimHandle sim);

/**
 * Get current scores.
 */
int ctf_sim_l_score(CTFSimHandle sim);
int ctf_sim_r_score(CTFSimHandle sim);

/**
 * Get step count.
 */
int ctf_sim_step_count(CTFSimHandle sim);

#ifdef __cplusplus
}
#endif

#endif /* CTF_SIM_H */
