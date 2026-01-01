/**
 * ctf_sim.c - CTF Game Simulator Implementation (Part 1: Data Structures)
 */

#include "ctf_sim.h"
#include "cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ============================================================
 * Constants
 * ============================================================ */
#define MAX_PLAYERS 6
#define MAX_FLAGS 32
#define MAX_BLOCKED 512
#define JSON_BUFFER_SIZE 16384
#define PRISON_DURATION_MS 20000

/* ============================================================
 * Data Structures
 * ============================================================ */

typedef struct {
    int x, y;
} Pos;

typedef struct {
    char name[8];       /* "L0", "L1", "L2", "R0", "R1", "R2" */
    char team;          /* 'L' or 'R' */
    Pos pos;
    Pos target_pos;
    int has_flag;
    int in_prison;
    int prison_time_left_ms;
    int prison_duration_ms;
} Player;

typedef struct {
    char team;          /* 'L' or 'R' */
    Pos pos;
    int can_pickup;
} Flag;

typedef struct {
    Pos tiles[9];       /* 3x3 grid */
} Zone;

typedef struct {
    int width;
    int height;
    int num_players;
    int num_flags;

    /* Random state (simple LCG) */
    unsigned int rng_state;

    /* Map */
    int* blocked;       /* blocked[y * width + x] = 1 if blocked */
    int num_blocked;
    Pos blocked_list[MAX_BLOCKED];

    /* Zones */
    Zone l_target;
    Zone l_prison;
    Zone r_target;
    Zone r_prison;

    /* Players */
    Player players[MAX_PLAYERS];
    int total_players;

    /* Flags */
    Flag flags[MAX_FLAGS];
    int flag_count;

    /* Scores */
    int l_score;
    int r_score;

    /* State */
    int step_count;
    double sim_time_ms;
    int done;

    /* Timing */
    int dt_ms;
    int move_duration_ms;
    int substep_ms;

    /* JSON output buffer */
    char json_buffer[JSON_BUFFER_SIZE];
} CTFSim;

/* ============================================================
 * Random Number Generator (Simple LCG)
 * ============================================================ */

static unsigned int rng_next(CTFSim* sim) {
    sim->rng_state = sim->rng_state * 1103515245 + 12345;
    return (sim->rng_state >> 16) & 0x7FFF;
}

static int rng_randint(CTFSim* sim, int min, int max) {
    if (min >= max) return min;
    return min + (rng_next(sim) % (max - min + 1));
}

/* ============================================================
 * Position Helpers
 * ============================================================ */

static int pos_eq(Pos a, Pos b) {
    return a.x == b.x && a.y == b.y;
}

static int in_bounds(CTFSim* sim, Pos p) {
    return p.x >= 0 && p.x < sim->width && p.y >= 0 && p.y < sim->height;
}

static int is_blocked(CTFSim* sim, Pos p) {
    if (!in_bounds(sim, p)) return 1;
    return sim->blocked[p.y * sim->width + p.x];
}

static void set_blocked(CTFSim* sim, Pos p) {
    if (in_bounds(sim, p)) {
        sim->blocked[p.y * sim->width + p.x] = 1;
        if (sim->num_blocked < MAX_BLOCKED) {
            sim->blocked_list[sim->num_blocked++] = p;
        }
    }
}

static Pos move_pos(Pos p, const char* action) {
    Pos result = p;
    if (action == NULL || action[0] == '\0') return result;
    if (strcmp(action, "up") == 0) { result.y--; }
    else if (strcmp(action, "down") == 0) { result.y++; }
    else if (strcmp(action, "left") == 0) { result.x--; }
    else if (strcmp(action, "right") == 0) { result.x++; }
    return result;
}

/* Create 3x3 zone centered at (cx, cy) */
static void create_zone(Zone* zone, int cx, int cy) {
    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            zone->tiles[idx].x = cx + dx;
            zone->tiles[idx].y = cy + dy;
            idx++;
        }
    }
}

static int pos_in_zone(Pos p, Zone* zone) {
    for (int i = 0; i < 9; i++) {
        if (pos_eq(p, zone->tiles[i])) return 1;
    }
    return 0;
}

/* ============================================================
 * Create / Destroy
 * ============================================================ */

CTFSimHandle ctf_sim_create(int width, int height, int num_players, int num_flags, int seed) {
    CTFSim* sim = (CTFSim*)calloc(1, sizeof(CTFSim));
    if (!sim) return NULL;

    sim->width = width > 0 ? width : 20;
    sim->height = height > 0 ? height : 20;
    sim->num_players = num_players > 0 ? num_players : 3;
    sim->num_flags = num_flags > 0 ? num_flags : 9;

    /* Initialize RNG */
    if (seed < 0) {
        sim->rng_state = (unsigned int)time(NULL);
    } else {
        sim->rng_state = (unsigned int)seed;
    }

    /* Allocate blocked map */
    sim->blocked = (int*)calloc(sim->width * sim->height, sizeof(int));
    if (!sim->blocked) {
        free(sim);
        return NULL;
    }

    /* Timing defaults */
    sim->dt_ms = 600;
    sim->move_duration_ms = 300;
    sim->substep_ms = 25;

    return (CTFSimHandle)sim;
}

void ctf_sim_destroy(CTFSimHandle handle) {
    if (!handle) return;
    CTFSim* sim = (CTFSim*)handle;
    if (sim->blocked) free(sim->blocked);
    free(sim);
}

/* ============================================================
 * Reset - Generate new game
 * ============================================================ */

const char* ctf_sim_reset(CTFSimHandle handle) {
    if (!handle) return NULL;
    CTFSim* sim = (CTFSim*)handle;

    /* Reset state */
    sim->step_count = 0;
    sim->sim_time_ms = 0.0;
    sim->done = 0;
    sim->l_score = 0;
    sim->r_score = 0;
    sim->flag_count = 0;
    sim->num_blocked = 0;

    /* Clear blocked map */
    memset(sim->blocked, 0, sim->width * sim->height * sizeof(int));

    /* Boundary walls */
    for (int x = 0; x < sim->width; x++) {
        set_blocked(sim, (Pos){x, 0});
        set_blocked(sim, (Pos){x, sim->height - 1});
    }
    for (int y = 0; y < sim->height; y++) {
        set_blocked(sim, (Pos){0, y});
        set_blocked(sim, (Pos){sim->width - 1, y});
    }

    /* Generate obstacles (8 single + 4 double) */
    Pos obstacles1[8];
    int obs1_count = 0;
    while (obs1_count < 8) {
        int x = rng_randint(sim, 4, sim->width - 5);
        int y = rng_randint(sim, 1, sim->height - 2);
        Pos p = {x, y};
        if (is_blocked(sim, p)) continue;
        int dup = 0;
        for (int i = 0; i < obs1_count; i++) {
            if (pos_eq(obstacles1[i], p)) { dup = 1; break; }
        }
        if (!dup) obstacles1[obs1_count++] = p;
    }

    Pos obstacles2[4];
    int obs2_count = 0;
    while (obs2_count < 4) {
        int x = rng_randint(sim, 4, sim->width - 5);
        int y = rng_randint(sim, 1, sim->height - 3);
        Pos p = {x, y};
        Pos p_below = {x, y + 1};
        Pos p_above = {x, y - 1};

        int skip = 0;
        for (int i = 0; i < obs1_count; i++) {
            if (pos_eq(obstacles1[i], p) || pos_eq(obstacles1[i], p_below)) {
                skip = 1; break;
            }
        }
        if (skip) continue;
        for (int i = 0; i < obs2_count; i++) {
            if (pos_eq(obstacles2[i], p) || pos_eq(obstacles2[i], p_above)) {
                skip = 1; break;
            }
        }
        if (!skip) obstacles2[obs2_count++] = p;
    }

    for (int i = 0; i < obs1_count; i++) set_blocked(sim, obstacles1[i]);
    for (int i = 0; i < obs2_count; i++) {
        set_blocked(sim, obstacles2[i]);
        set_blocked(sim, (Pos){obstacles2[i].x, obstacles2[i].y + 1});
    }

    /* Create zones */
    int center_y = sim->height / 2;
    create_zone(&sim->l_target, 2, center_y);
    create_zone(&sim->l_prison, 2, sim->height - 3);
    create_zone(&sim->r_target, sim->width - 3, center_y);
    create_zone(&sim->r_prison, sim->width - 3, sim->height - 3);

    /* Generate flags */
    sim->flag_count = 0;
    int l_flag_count = 0;
    while (l_flag_count < sim->num_flags && sim->flag_count < MAX_FLAGS) {
        int x = rng_randint(sim, 2, (sim->width / 2) - 1);
        int y = rng_randint(sim, 1, sim->height - 3);
        Pos p = {x, y};
        if (is_blocked(sim, p)) continue;
        int dup = 0;
        for (int i = 0; i < sim->flag_count; i++) {
            if (pos_eq(sim->flags[i].pos, p)) { dup = 1; break; }
        }
        if (!dup) {
            sim->flags[sim->flag_count].team = 'L';
            sim->flags[sim->flag_count].pos = p;
            sim->flags[sim->flag_count].can_pickup = 1;
            sim->flag_count++;
            l_flag_count++;
        }
    }

    int r_flag_count = 0;
    while (r_flag_count < sim->num_flags && sim->flag_count < MAX_FLAGS) {
        int x = rng_randint(sim, sim->width / 2, sim->width - 2);
        int y = rng_randint(sim, 1, sim->height - 3);
        Pos p = {x, y};
        if (is_blocked(sim, p)) continue;
        int dup = 0;
        for (int i = 0; i < sim->flag_count; i++) {
            if (pos_eq(sim->flags[i].pos, p)) { dup = 1; break; }
        }
        if (!dup) {
            sim->flags[sim->flag_count].team = 'R';
            sim->flags[sim->flag_count].pos = p;
            sim->flags[sim->flag_count].can_pickup = 1;
            sim->flag_count++;
            r_flag_count++;
        }
    }

    /* Initialize players */
    sim->total_players = sim->num_players * 2;
    int l_px = 1, r_px = sim->width - 2;

    for (int i = 0; i < sim->num_players; i++) {
        Player* p = &sim->players[i];
        snprintf(p->name, sizeof(p->name), "L%d", i);
        p->team = 'L';
        p->pos = (Pos){l_px, i + 1};
        p->target_pos = p->pos;
        p->has_flag = 0;
        p->in_prison = 0;
        p->prison_time_left_ms = 0;
        p->prison_duration_ms = PRISON_DURATION_MS;
    }

    for (int i = 0; i < sim->num_players; i++) {
        Player* p = &sim->players[sim->num_players + i];
        snprintf(p->name, sizeof(p->name), "R%d", i);
        p->team = 'R';
        p->pos = (Pos){r_px, i + 1};
        p->target_pos = p->pos;
        p->has_flag = 0;
        p->in_prison = 0;
        p->prison_time_left_ms = 0;
        p->prison_duration_ms = PRISON_DURATION_MS;
    }

    return "ok";
}

/* ============================================================
 * Prison Helpers
 * ============================================================ */

static Pos find_available_prison_tile(CTFSim* sim, Zone* prison, char team) {
    for (int i = 0; i < 9; i++) {
        Pos tile = prison->tiles[i];
        int occupied = 0;
        for (int j = 0; j < sim->total_players; j++) {
            Player* p = &sim->players[j];
            if (p->team == team && p->in_prison && pos_eq(p->pos, tile)) {
                occupied = 1;
                break;
            }
        }
        if (!occupied) return tile;
    }
    return prison->tiles[0];
}

static void send_to_prison(CTFSim* sim, Player* player, Pos capture_pos) {
    /* Drop flag at capture position */
    if (player->has_flag) {
        char flag_team = (player->team == 'L') ? 'R' : 'L';
        if (sim->flag_count < MAX_FLAGS) {
            sim->flags[sim->flag_count].team = flag_team;
            sim->flags[sim->flag_count].pos = capture_pos;
            sim->flags[sim->flag_count].can_pickup = 1;
            sim->flag_count++;
        }
        player->has_flag = 0;
    }

    /* Find prison spot */
    Zone* prison = (player->team == 'L') ? &sim->l_prison : &sim->r_prison;
    Pos spot = find_available_prison_tile(sim, prison, player->team);

    player->pos = spot;
    player->target_pos = spot;
    player->in_prison = 1;
    player->prison_time_left_ms = player->prison_duration_ms;
}

/* ============================================================
 * Step - Main game loop
 * ============================================================ */

const char* ctf_sim_step(CTFSimHandle handle, const char* actions_json) {
    if (!handle) return NULL;
    CTFSim* sim = (CTFSim*)handle;
    if (sim->done) return "done";

    /* Parse actions JSON */
    cJSON* actions = cJSON_Parse(actions_json);
    if (!actions) return "error: invalid json";

    /* Record start positions and determine targets */
    Pos start_pos[MAX_PLAYERS];
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        start_pos[i] = p->pos;

        if (p->in_prison) {
            p->target_pos = p->pos;
            continue;
        }

        cJSON* act_item = cJSON_GetObjectItem(actions, p->name);
        const char* act = (act_item && cJSON_IsString(act_item)) ? act_item->valuestring : "";

        Pos nxt = move_pos(p->pos, act);
        if (!in_bounds(sim, nxt) || is_blocked(sim, nxt)) {
            nxt = p->pos;
        }
        p->target_pos = nxt;
    }

    /* Advance time and prison timers */
    int remaining = sim->move_duration_ms;
    while (remaining > 0) {
        int dt = (sim->substep_ms < remaining) ? sim->substep_ms : remaining;
        for (int i = 0; i < sim->total_players; i++) {
            Player* p = &sim->players[i];
            if (p->in_prison) {
                p->prison_time_left_ms -= dt;
                if (p->prison_time_left_ms <= 0) {
                    p->prison_time_left_ms = 0;
                    p->in_prison = 0;
                }
            }
        }
        sim->sim_time_ms += dt;
        remaining -= dt;
    }

    cJSON_Delete(actions);

    /* Collision detection */
    double middle_line = sim->width / 2.0;
    int captured[MAX_PLAYERS] = {0};
    Pos capture_pos[MAX_PLAYERS];

    for (int i = 0; i < sim->total_players; i++) {
        Player* a = &sim->players[i];
        if (a->in_prison) continue;

        for (int j = i + 1; j < sim->total_players; j++) {
            Player* b = &sim->players[j];
            if (b->in_prison) continue;
            if (a->team == b->team) continue;

            Pos a_start = start_pos[i], a_end = a->target_pos;
            Pos b_start = start_pos[j], b_end = b->target_pos;

            Pos collision_tile = {-1, -1};
            double collision_mid_x = -1;

            /* Same destination */
            if (pos_eq(a_end, b_end)) {
                collision_tile = a_end;
                collision_mid_x = (double)collision_tile.x;
            }
            /* Head-on collision */
            else if (pos_eq(a_end, b_start) && pos_eq(b_end, a_start)) {
                collision_mid_x = (a_start.x + a_end.x) / 2.0;
                int mid_x = (int)(collision_mid_x + 0.5);
                int mid_y = (int)((a_start.y + a_end.y) / 2.0 + 0.5);
                collision_tile = (Pos){mid_x, mid_y};
            }
            /* A moves into stationary B */
            else if (pos_eq(a_end, b_start) && pos_eq(b_end, b_start)) {
                collision_tile = b_start;
                collision_mid_x = (double)collision_tile.x;
            }
            /* B moves into stationary A */
            else if (pos_eq(b_end, a_start) && pos_eq(a_end, a_start)) {
                collision_tile = a_start;
                collision_mid_x = (double)collision_tile.x;
            }

            if (collision_tile.x < 0) continue;

            int left_half = collision_mid_x < middle_line;
            int caught_idx = left_half ? j : i;

            if (!captured[caught_idx]) {
                captured[caught_idx] = 1;
                capture_pos[caught_idx] = collision_tile;
            }
        }
    }

    /* Send captured players to prison */
    for (int i = 0; i < sim->total_players; i++) {
        if (captured[i]) {
            send_to_prison(sim, &sim->players[i], capture_pos[i]);
        }
    }

    /* Update positions for non-captured players */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison) {
            p->target_pos = p->pos;
            continue;
        }
        if (!captured[i]) {
            p->pos = p->target_pos;
        }
    }

    /* Flag pickup */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison || p->has_flag) continue;

        for (int f = 0; f < sim->flag_count; f++) {
            Flag* flag = &sim->flags[f];
            if (flag->team == p->team) continue;
            if (!flag->can_pickup) continue;
            if (pos_eq(flag->pos, p->pos)) {
                p->has_flag = 1;
                /* Remove flag by shifting */
                for (int k = f; k < sim->flag_count - 1; k++) {
                    sim->flags[k] = sim->flags[k + 1];
                }
                sim->flag_count--;
                break;
            }
        }
    }

    /* Score flags */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison || !p->has_flag) continue;

        Zone* target = (p->team == 'L') ? &sim->l_target : &sim->r_target;
        if (pos_in_zone(p->pos, target)) {
            p->has_flag = 0;
            if (p->team == 'L') sim->l_score++;
            else sim->r_score++;

            /* Spawn new flag in target zone */
            char enemy_team = (p->team == 'L') ? 'R' : 'L';
            if (sim->flag_count < MAX_FLAGS) {
                for (int t = 0; t < 9; t++) {
                    Pos tile = target->tiles[t];
                    int occupied = 0;
                    for (int ff = 0; ff < sim->flag_count; ff++) {
                        if (sim->flags[ff].team == enemy_team &&
                            pos_eq(sim->flags[ff].pos, tile) &&
                            !sim->flags[ff].can_pickup) {
                            occupied = 1;
                            break;
                        }
                    }
                    if (!occupied) {
                        sim->flags[sim->flag_count].team = enemy_team;
                        sim->flags[sim->flag_count].pos = tile;
                        sim->flags[sim->flag_count].can_pickup = 0;
                        sim->flag_count++;
                        break;
                    }
                }
            }
        }
    }

    /* Rescue */
    for (int team_idx = 0; team_idx < 2; team_idx++) {
        char team = (team_idx == 0) ? 'L' : 'R';
        Zone* prison = (team == 'L') ? &sim->l_prison : &sim->r_prison;

        int rescuer_present = 0;
        for (int i = 0; i < sim->total_players; i++) {
            Player* p = &sim->players[i];
            if (p->team == team && !p->in_prison && pos_in_zone(p->pos, prison)) {
                rescuer_present = 1;
                break;
            }
        }

        if (rescuer_present) {
            for (int i = 0; i < sim->total_players; i++) {
                Player* p = &sim->players[i];
                if (p->team == team && p->in_prison) {
                    p->in_prison = 0;
                }
            }
        }
    }

    /* End game check */
    sim->step_count++;
    if (sim->l_score >= sim->num_flags || sim->r_score >= sim->num_flags) {
        sim->done = 1;
    }

    return "ok";
}

/* ============================================================
 * Step Binary - Fast version without JSON parsing
 * ============================================================ */

static const char* action_from_code(unsigned char code) {
    switch (code) {
        case 1: return "up";
        case 2: return "down";
        case 3: return "left";
        case 4: return "right";
        default: return "";
    }
}

const char* ctf_sim_step_binary(CTFSimHandle handle, const unsigned char* actions) {
    if (!handle || !actions) return NULL;
    CTFSim* sim = (CTFSim*)handle;
    if (sim->done) return "done";

    /* Record start positions and determine targets */
    Pos start_pos[MAX_PLAYERS];
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        start_pos[i] = p->pos;

        if (p->in_prison) {
            p->target_pos = p->pos;
            continue;
        }

        /* Get action from binary array: [L0, L1, L2, R0, R1, R2] */
        const char* act = action_from_code(actions[i]);

        Pos nxt = move_pos(p->pos, act);
        if (!in_bounds(sim, nxt) || is_blocked(sim, nxt)) {
            nxt = p->pos;
        }
        p->target_pos = nxt;
    }

    /* Advance time and prison timers */
    int remaining = sim->move_duration_ms;
    while (remaining > 0) {
        int dt = (sim->substep_ms < remaining) ? sim->substep_ms : remaining;
        for (int i = 0; i < sim->total_players; i++) {
            Player* p = &sim->players[i];
            if (p->in_prison) {
                p->prison_time_left_ms -= dt;
                if (p->prison_time_left_ms <= 0) {
                    p->prison_time_left_ms = 0;
                    p->in_prison = 0;
                }
            }
        }
        sim->sim_time_ms += dt;
        remaining -= dt;
    }

    /* Collision detection */
    double middle_line = sim->width / 2.0;
    int captured[MAX_PLAYERS] = {0};
    Pos capture_pos[MAX_PLAYERS];

    for (int i = 0; i < sim->total_players; i++) {
        Player* a = &sim->players[i];
        if (a->in_prison) continue;

        for (int j = i + 1; j < sim->total_players; j++) {
            Player* b = &sim->players[j];
            if (b->in_prison) continue;
            if (a->team == b->team) continue;

            Pos a_start = start_pos[i], a_end = a->target_pos;
            Pos b_start = start_pos[j], b_end = b->target_pos;

            Pos collision_tile = {-1, -1};
            double collision_mid_x = -1;

            if (pos_eq(a_end, b_end)) {
                collision_tile = a_end;
                collision_mid_x = (double)collision_tile.x;
            }
            else if (pos_eq(a_end, b_start) && pos_eq(b_end, a_start)) {
                collision_mid_x = (a_start.x + a_end.x) / 2.0;
                int mid_x = (int)(collision_mid_x + 0.5);
                int mid_y = (int)((a_start.y + a_end.y) / 2.0 + 0.5);
                collision_tile = (Pos){mid_x, mid_y};
            }
            else if (pos_eq(a_end, b_start) && pos_eq(b_end, b_start)) {
                collision_tile = b_start;
                collision_mid_x = (double)collision_tile.x;
            }
            else if (pos_eq(b_end, a_start) && pos_eq(a_end, a_start)) {
                collision_tile = a_start;
                collision_mid_x = (double)collision_tile.x;
            }

            if (collision_tile.x < 0) continue;

            int left_half = collision_mid_x < middle_line;
            int caught_idx = left_half ? j : i;

            if (!captured[caught_idx]) {
                captured[caught_idx] = 1;
                capture_pos[caught_idx] = collision_tile;
            }
        }
    }

    /* Send captured players to prison */
    for (int i = 0; i < sim->total_players; i++) {
        if (captured[i]) {
            send_to_prison(sim, &sim->players[i], capture_pos[i]);
        }
    }

    /* Update positions for non-captured players */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison) {
            p->target_pos = p->pos;
            continue;
        }
        if (!captured[i]) {
            p->pos = p->target_pos;
        }
    }

    /* Flag pickup */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison || p->has_flag) continue;

        for (int f = 0; f < sim->flag_count; f++) {
            Flag* flag = &sim->flags[f];
            if (flag->team == p->team) continue;
            if (!flag->can_pickup) continue;
            if (pos_eq(flag->pos, p->pos)) {
                p->has_flag = 1;
                for (int k = f; k < sim->flag_count - 1; k++) {
                    sim->flags[k] = sim->flags[k + 1];
                }
                sim->flag_count--;
                break;
            }
        }
    }

    /* Score flags */
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->in_prison || !p->has_flag) continue;

        Zone* target = (p->team == 'L') ? &sim->l_target : &sim->r_target;
        if (pos_in_zone(p->pos, target)) {
            p->has_flag = 0;
            if (p->team == 'L') sim->l_score++;
            else sim->r_score++;

            char enemy_team = (p->team == 'L') ? 'R' : 'L';
            if (sim->flag_count < MAX_FLAGS) {
                for (int t = 0; t < 9; t++) {
                    Pos tile = target->tiles[t];
                    int occupied = 0;
                    for (int ff = 0; ff < sim->flag_count; ff++) {
                        if (sim->flags[ff].team == enemy_team &&
                            pos_eq(sim->flags[ff].pos, tile) &&
                            !sim->flags[ff].can_pickup) {
                            occupied = 1;
                            break;
                        }
                    }
                    if (!occupied) {
                        sim->flags[sim->flag_count].team = enemy_team;
                        sim->flags[sim->flag_count].pos = tile;
                        sim->flags[sim->flag_count].can_pickup = 0;
                        sim->flag_count++;
                        break;
                    }
                }
            }
        }
    }

    /* Rescue */
    for (int team_idx = 0; team_idx < 2; team_idx++) {
        char team = (team_idx == 0) ? 'L' : 'R';
        Zone* prison = (team == 'L') ? &sim->l_prison : &sim->r_prison;

        int rescuer_present = 0;
        for (int i = 0; i < sim->total_players; i++) {
            Player* p = &sim->players[i];
            if (p->team == team && !p->in_prison && pos_in_zone(p->pos, prison)) {
                rescuer_present = 1;
                break;
            }
        }

        if (rescuer_present) {
            for (int i = 0; i < sim->total_players; i++) {
                Player* p = &sim->players[i];
                if (p->team == team && p->in_prison) {
                    p->in_prison = 0;
                }
            }
        }
    }

    /* End game check */
    sim->step_count++;
    if (sim->l_score >= sim->num_flags || sim->r_score >= sim->num_flags) {
        sim->done = 1;
    }

    return "ok";
}

/* ============================================================
 * Status - Generate JSON status for a team
 * ============================================================ */

static void add_zone_to_json(cJSON* parent, const char* name, Zone* zone) {
    cJSON* arr = cJSON_CreateArray();
    for (int i = 0; i < 9; i++) {
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddNumberToObject(obj, "x", zone->tiles[i].x);
        cJSON_AddNumberToObject(obj, "y", zone->tiles[i].y);
        cJSON_AddItemToArray(arr, obj);
    }
    cJSON_AddItemToObject(parent, name, arr);
}

const char* ctf_sim_status(CTFSimHandle handle, const char* team) {
    if (!handle || !team) return NULL;
    CTFSim* sim = (CTFSim*)handle;

    char my_team = team[0];
    char opp_team = (my_team == 'L') ? 'R' : 'L';

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "action", "status");
    cJSON_AddNumberToObject(root, "time", sim->sim_time_ms);

    /* My players */
    cJSON* my_players = cJSON_CreateArray();
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->team != my_team) continue;
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddStringToObject(obj, "name", p->name);
        char team_str[2] = {p->team, '\0'};
        cJSON_AddStringToObject(obj, "team", team_str);
        cJSON_AddBoolToObject(obj, "hasFlag", p->has_flag);
        cJSON_AddNumberToObject(obj, "posX", p->target_pos.x);
        cJSON_AddNumberToObject(obj, "posY", p->target_pos.y);
        cJSON_AddBoolToObject(obj, "inPrison", p->in_prison);
        cJSON_AddNumberToObject(obj, "inPrisonTimeLeft", p->prison_time_left_ms);
        cJSON_AddNumberToObject(obj, "inPrisonDuration", p->prison_duration_ms);
        cJSON_AddNumberToObject(obj, "_tileX", p->pos.x);
        cJSON_AddNumberToObject(obj, "_tileY", p->pos.y);
        cJSON_AddItemToArray(my_players, obj);
    }
    cJSON_AddItemToObject(root, "myteamPlayer", my_players);

    /* Opponent players */
    cJSON* opp_players = cJSON_CreateArray();
    for (int i = 0; i < sim->total_players; i++) {
        Player* p = &sim->players[i];
        if (p->team != opp_team) continue;
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddStringToObject(obj, "name", p->name);
        char team_str[2] = {p->team, '\0'};
        cJSON_AddStringToObject(obj, "team", team_str);
        cJSON_AddBoolToObject(obj, "hasFlag", p->has_flag);
        cJSON_AddNumberToObject(obj, "posX", p->target_pos.x);
        cJSON_AddNumberToObject(obj, "posY", p->target_pos.y);
        cJSON_AddBoolToObject(obj, "inPrison", p->in_prison);
        cJSON_AddNumberToObject(obj, "inPrisonTimeLeft", p->prison_time_left_ms);
        cJSON_AddNumberToObject(obj, "inPrisonDuration", p->prison_duration_ms);
        cJSON_AddNumberToObject(obj, "_tileX", p->pos.x);
        cJSON_AddNumberToObject(obj, "_tileY", p->pos.y);
        cJSON_AddItemToArray(opp_players, obj);
    }
    cJSON_AddItemToObject(root, "opponentPlayer", opp_players);

    /* My flags */
    cJSON* my_flags = cJSON_CreateArray();
    for (int i = 0; i < sim->flag_count; i++) {
        if (sim->flags[i].team != my_team) continue;
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddBoolToObject(obj, "canPickup", sim->flags[i].can_pickup);
        cJSON_AddNumberToObject(obj, "posX", sim->flags[i].pos.x);
        cJSON_AddNumberToObject(obj, "posY", sim->flags[i].pos.y);
        cJSON_AddItemToArray(my_flags, obj);
    }
    cJSON_AddItemToObject(root, "myteamFlag", my_flags);

    /* Opponent flags */
    cJSON* opp_flags = cJSON_CreateArray();
    for (int i = 0; i < sim->flag_count; i++) {
        if (sim->flags[i].team != opp_team) continue;
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddBoolToObject(obj, "canPickup", sim->flags[i].can_pickup);
        cJSON_AddNumberToObject(obj, "posX", sim->flags[i].pos.x);
        cJSON_AddNumberToObject(obj, "posY", sim->flags[i].pos.y);
        cJSON_AddItemToArray(opp_flags, obj);
    }
    cJSON_AddItemToObject(root, "opponentFlag", opp_flags);

    /* Scores */
    int my_score = (my_team == 'L') ? sim->l_score : sim->r_score;
    int opp_score = (my_team == 'L') ? sim->r_score : sim->l_score;
    cJSON_AddNumberToObject(root, "myteamScore", my_score);
    cJSON_AddNumberToObject(root, "opponentScore", opp_score);

    /* Zones */
    Zone* my_target = (my_team == 'L') ? &sim->l_target : &sim->r_target;
    Zone* my_prison = (my_team == 'L') ? &sim->l_prison : &sim->r_prison;
    Zone* opp_target = (my_team == 'L') ? &sim->r_target : &sim->l_target;
    Zone* opp_prison = (my_team == 'L') ? &sim->r_prison : &sim->l_prison;

    add_zone_to_json(root, "_myteamTarget", my_target);
    add_zone_to_json(root, "_myteamPrison", my_prison);
    add_zone_to_json(root, "_opponentTarget", opp_target);
    add_zone_to_json(root, "_opponentPrison", opp_prison);

    /* Output to buffer */
    char* json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        strncpy(sim->json_buffer, json_str, JSON_BUFFER_SIZE - 1);
        sim->json_buffer[JSON_BUFFER_SIZE - 1] = '\0';
        free(json_str);
    }
    cJSON_Delete(root);

    return sim->json_buffer;
}

/* ============================================================
 * Init Payload
 * ============================================================ */

const char* ctf_sim_init_payload(CTFSimHandle handle, const char* team) {
    if (!handle || !team) return NULL;
    CTFSim* sim = (CTFSim*)handle;

    char my_team = team[0];

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "action", "init");

    /* Map */
    cJSON* map = cJSON_CreateObject();
    cJSON_AddNumberToObject(map, "width", sim->width);
    cJSON_AddNumberToObject(map, "height", sim->height);

    cJSON* walls = cJSON_CreateArray();
    for (int i = 0; i < sim->num_blocked; i++) {
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddNumberToObject(obj, "x", sim->blocked_list[i].x);
        cJSON_AddNumberToObject(obj, "y", sim->blocked_list[i].y);
        cJSON_AddItemToArray(walls, obj);
    }
    cJSON_AddItemToObject(map, "walls", walls);
    cJSON_AddItemToObject(map, "obstacles", cJSON_CreateArray());
    cJSON_AddItemToObject(root, "map", map);

    cJSON_AddNumberToObject(root, "numPlayers", sim->num_players);
    cJSON_AddNumberToObject(root, "numFlags", sim->num_flags);

    char team_str[2] = {my_team, '\0'};
    cJSON_AddStringToObject(root, "myteamName", team_str);

    Zone* my_target = (my_team == 'L') ? &sim->l_target : &sim->r_target;
    Zone* my_prison = (my_team == 'L') ? &sim->l_prison : &sim->r_prison;
    Zone* opp_target = (my_team == 'L') ? &sim->r_target : &sim->l_target;
    Zone* opp_prison = (my_team == 'L') ? &sim->r_prison : &sim->l_prison;

    add_zone_to_json(root, "myteamTarget", my_target);
    add_zone_to_json(root, "myteamPrison", my_prison);
    add_zone_to_json(root, "opponentTarget", opp_target);
    add_zone_to_json(root, "opponentPrison", opp_prison);

    char* json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        strncpy(sim->json_buffer, json_str, JSON_BUFFER_SIZE - 1);
        sim->json_buffer[JSON_BUFFER_SIZE - 1] = '\0';
        free(json_str);
    }
    cJSON_Delete(root);

    return sim->json_buffer;
}

/* ============================================================
 * Simple Getters
 * ============================================================ */

int ctf_sim_done(CTFSimHandle handle) {
    if (!handle) return 1;
    return ((CTFSim*)handle)->done;
}

int ctf_sim_l_score(CTFSimHandle handle) {
    if (!handle) return 0;
    return ((CTFSim*)handle)->l_score;
}

int ctf_sim_r_score(CTFSimHandle handle) {
    if (!handle) return 0;
    return ((CTFSim*)handle)->r_score;
}

int ctf_sim_step_count(CTFSimHandle handle) {
    if (!handle) return 0;
    return ((CTFSim*)handle)->step_count;
}
