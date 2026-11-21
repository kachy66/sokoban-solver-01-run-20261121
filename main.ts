// Push-optimal Sokoban solver.
//
// Strategy, written from scratch for this bounty:
//  * A level is a hallway with one to four small rooms off it, and a box
//    pushed out of its room can never come back, so each room is solved on
//    its own and the answers are stitched together with plain walks.
//  * Inside a room the search is breadth-first over pushes (so the first
//    solved state found uses the fewest pushes), with two sound prunes:
//    a box standing on a square from which no sequence of pushes can ever
//    reach a goal (found by pulling boxes back from the goals), and a box
//    caught in a frozen 2x2 block of walls and boxes with a box off a goal.
//  * A state's identity is its boxes plus the lowest square of the region
//    the player can walk to, so positions that differ only by where the
//    player stands inside one region count as one state.
//  * If a room's breadth-first search runs out of its state budget, a
//    push-based A* with an assignment lower bound tries to solve it anyway
//    (more pushes than optimal, but a solved room is worth far more than
//    an unsolved one); if the level has no clean room split, the same
//    search runs over the whole level.
//  * Everything is budgeted by states counted, never by the clock, and the
//    search order is fixed, so two runs write identical bytes.
//
// Usage: node main.ts, reading /input/<id>/level.txt (or MAKEFUN_INPUT_DIR)
// and writing /output/<id>/solution.txt (or MAKEFUN_OUTPUT_DIR). Node 22,
// standard library only, no network, no clock reads.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, in states created. Work, never the clock: the same input
// always costs the same work, so the output is byte-for-byte repeatable.
const ROOM_BFS_CAP = 500_000;
const LEVEL_BFS_CAP = 1_000_000;
const GREEDY_CAP = 500_000;
// One budget for the whole run, so even a set of rooms that all run out
// of their own budgets cannot push the run past the 60-second wall.
const WORK_CAP = 6_000_000;
let workUsed = 0;

type Dir = { dx: number; dy: number; ch: string };
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

type Board = {
  w: number;
  h: number;
  size: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
};

type Answer = { pushes: number; moves: string } | null;

/** Parse XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goalSet = new Uint8Array(size);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, size, wall, goals, goalSet, boxes, player };
}

/**
 * The squares a box can stand on and still be pushed onto some goal:
 * start from the goals and pull a box backwards, one step at a time, over
 * open floor (the square the player would have to stand on must be open
 * too). A box anywhere else is dead and can be pruned outright.
 */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.size);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const step = d.dy * b.w + d.dx;
      const y = x - step;
      const behind = y - step;
      if (y < 0 || y >= b.size || behind < 0 || behind >= b.size) continue;
      if (b.wall[y] === 0 && b.wall[behind] === 0 && live[y] === 0) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood-fill the squares the player can walk to from `from` (walls and
 * boxes in `occ` block it), marking them with `stamp` in `seen`, and
 * return the lowest such square. `stack` is scratch space of board size.
 */
function walkRegion(
  b: Board,
  occ: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const q0 = p - b.w;
    const q1 = p + b.w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (b.wall[q0] === 0 && occ[q0] === 0 && seen[q0] !== stamp) {
      seen[q0] = stamp;
      stack[n++] = q0;
    }
    if (b.wall[q1] === 0 && occ[q1] === 0 && seen[q1] !== stamp) {
      seen[q1] = stamp;
      stack[n++] = q1;
    }
    if (b.wall[q2] === 0 && occ[q2] === 0 && seen[q2] !== stamp) {
      seen[q2] = stamp;
      stack[n++] = q2;
    }
    if (b.wall[q3] === 0 && occ[q3] === 0 && seen[q3] !== stamp) {
      seen[q3] = stamp;
      stack[n++] = q3;
    }
  }
  return low;
}

/**
 * True when the box just pushed onto `t` is stuck in a 2x2 block made of
 * walls and boxes that holds a box off its goal: no push can ever move
 * any box of the block, so the position is dead.
 */
function frozenBlock(b: Board, occ: Uint8Array, t: number): boolean {
  const corners = [t, t - 1, t - b.w, t - b.w - 1];
  for (const a of corners) {
    if (a < 0 || a % b.w >= b.w - 1 || a + b.w >= b.size) continue;
    const c = a + b.w;
    const d = c + 1;
    const e = a + 1;
    if (
      (b.wall[a] || occ[a]) &&
      (b.wall[e] || occ[e]) &&
      (b.wall[c] || occ[c]) &&
      (b.wall[d] || occ[d])
    ) {
      if (
        (occ[a] && !b.goalSet[a]) ||
        (occ[e] && !b.goalSet[e]) ||
        (occ[c] && !b.goalSet[c]) ||
        (occ[d] && !b.goalSet[d])
      ) {
        return true;
      }
    }
  }
  return false;
}

/** A state's key: the boxes (sorted) plus its walk region's lowest square. */
function stateKey(boxes: number[], low: number, size: number): number | string {
  if (boxes.length <= 4) {
    let k = low;
    for (const x of boxes) k = k * size + x;
    return k;
  }
  return `${low}:${boxes.join(',')}`;
}

/** Whether every box stands on a goal. */
function solved(goalSet: Uint8Array, boxes: number[]): boolean {
  for (const x of boxes) if (!goalSet[x]) return false;
  return true;
}

/**
 * The shortest walk from `from` to `to` around walls and boxes in `occ`,
 * as lower-case move letters, in direction order u, d, l, r. Empty when
 * the two squares are the same.
 */
function walkTo(b: Board, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.size).fill(-1);
  const how = new Array<string>(b.size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * b.w + d.dx;
      if (b.wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.ch;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walk target out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/**
 * Breadth-first search over pushes, which finds a solution with the fewest
 * pushes when one exists within `cap` states. States are kept in the order
 * found, with a parent link and the push that made them, so the answer is
 * rebuilt by walking the chain back to the start and filling in the walks
 * between pushes.
 */
function solvePushBfs(board: Board, cap: number): Answer {
  const live = liveSquares(board);
  for (const x of board.boxes) if (!live[x]) return null;
  const occ = new Uint8Array(board.size);
  const seen = new Int32Array(board.size);
  const mark = new Int32Array(board.size);
  const stack = new Int32Array(board.size);
  let stamp = 0;

  const stBoxes: number[][] = [board.boxes.slice()];
  const stPlayer: number[] = [board.player];
  const stParent: number[] = [-1];
  const stDir: number[] = [-1];
  const index = new Map<number | string, number>();
  for (const x of board.boxes) occ[x] = 1;
  const low0 = walkRegion(board, occ, board.player, seen, ++stamp, stack);
  index.set(stateKey(board.boxes, low0, board.size), 0);
  for (const x of board.boxes) occ[x] = 0;
  if (solved(board.goalSet, board.boxes)) return { pushes: 0, moves: '' };

  let found = -1;
  for (let head = 0; head < stBoxes.length && found < 0; head++) {
    const boxes = stBoxes[head]!;
    for (const x of boxes) occ[x] = 1;
    walkRegion(board, occ, stPlayer[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const from = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]!;
        const step = d.dy * board.w + d.dx;
        const to = from + step;
        // The player must be able to stand behind the box, and the square
        // beyond it must be open floor and still live for a box.
        if (seen[from - step] !== here || board.wall[to] || occ[to] || !live[to]) continue;
        occ[from] = 0;
        occ[to] = 1;
        const frozen = frozenBlock(board, occ, to);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = to;
          next.sort((a, b2) => a - b2);
          key = stateKey(
            next,
            walkRegion(board, occ, from, mark, ++stamp, stack),
            board.size,
          );
        }
        occ[to] = 0;
        occ[from] = 1;
        if (frozen || index.has(key)) continue;
        if (stBoxes.length >= cap || workUsed >= WORK_CAP) return null;
        workUsed++;
        index.set(key, stBoxes.length);
        stBoxes.push(next!);
        stPlayer.push(from);
        stParent.push(head);
        stDir.push(k);
        if (solved(board.goalSet, next!)) {
          found = stBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = stParent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  const boxes = board.boxes.slice();
  for (const s of chain) {
    const prevBoxes = stBoxes[stParent[s]!]!;
    const curBoxes = stBoxes[s]!;
    const from = stPlayer[s]!;
    let to = -1;
    for (const x of curBoxes) if (!prevBoxes.includes(x)) { to = x; break; }
    const k = stDir[s]!;
    const d = DIRS[k]!;
    const step = d.dy * board.w + d.dx;
    for (const x of boxes) occ[x] = 1;
    moves += walkTo(board, occ, player, from - step);
    for (const x of boxes) occ[x] = 0;
    moves += d.ch;
    player = from;
    boxes[boxes.indexOf(from)] = to;
    boxes.sort((a, b2) => a - b2);
  }
  return { pushes: chain.length, moves };
}

/**
 * Push-based A* fallback: same states and prunes as the breadth-first
 * search, ordered by pushes made plus a lower bound on the pushes still
 * needed (the cheapest assignment of boxes to goals, each box needing at
 * least its open-floor distance to its goal). Not guaranteed optimal, but
 * it still solves rooms the breadth-first search gave up on.
 */
function solvePushGreedy(board: Board, cap: number): Answer {
  const live = liveSquares(board);
  for (const x of board.boxes) if (!live[x]) return null;
  const n = board.boxes.length;
  // Distance from every square to every goal, over open floor.
  const dist: Int32Array[] = [];
  for (const g of board.goals) {
    const dd = new Int32Array(board.size).fill(-1);
    const q: number[] = [g];
    dd[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const d of DIRS) {
        const t = p + d.dy * board.w + d.dx;
        if (board.wall[t] || dd[t] !== -1) continue;
        dd[t] = dd[p]! + 1;
        q.push(t);
      }
    }
    dist.push(dd);
  }
  const lowerBound = (boxes: number[]): number => {
    // Cheapest assignment of boxes to distinct goals, by brute force
    // (at most 4 boxes, so at most 24 orders).
    const order = boxes.map((_, i) => i);
    let best = Infinity;
    const permute = (i: number): void => {
      if (i === n) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          const d = dist[j]![boxes[order[j]!]!]!;
          if (d < 0) return;
          sum += d;
        }
        if (sum < best) best = sum;
        return;
      }
      for (let j = i; j < n; j++) {
        [order[i], order[j]] = [order[j]!, order[i]!];
        permute(i + 1);
        [order[i], order[j]] = [order[j]!, order[i]!];
      }
    };
    permute(0);
    return best;
  };

  const occ = new Uint8Array(board.size);
  const seen = new Int32Array(board.size);
  const mark = new Int32Array(board.size);
  const stack = new Int32Array(board.size);
  let stamp = 0;
  const stBoxes: number[][] = [board.boxes.slice()];
  const stPlayer: number[] = [board.player];
  const stParent: number[] = [-1];
  const stDir: number[] = [-1];
  const stG: number[] = [0];
  const bestG = new Map<number | string, number>();
  for (const x of board.boxes) occ[x] = 1;
  const low0 = walkRegion(board, occ, board.player, seen, ++stamp, stack);
  bestG.set(stateKey(board.boxes, low0, board.size), 0);
  for (const x of board.boxes) occ[x] = 0;
  if (solved(board.goalSet, board.boxes)) return { pushes: 0, moves: '' };

  // Binary heap on [f, g, state], smallest f first.
  const heap: [number, number, number][] = [[lowerBound(board.boxes), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let found = -1;
  while (heap.length > 0 && found < 0) {
    const [, g, s] = heapPop();
    const boxes = stBoxes[s]!;
    if (g !== stG[s]) continue;
    for (const x of boxes) occ[x] = 1;
    walkRegion(board, occ, stPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const from = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]!;
        const step = d.dy * board.w + d.dx;
        const to = from + step;
        if (seen[from - step] !== here || board.wall[to] || occ[to] || !live[to]) continue;
        occ[from] = 0;
        occ[to] = 1;
        const frozen = frozenBlock(board, occ, to);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = to;
          next.sort((a, b2) => a - b2);
          key = stateKey(
            next,
            walkRegion(board, occ, from, mark, ++stamp, stack),
            board.size,
          );
        }
        occ[to] = 0;
        occ[from] = 1;
        if (frozen) continue;
        const known = bestG.get(key);
        if (known !== undefined && known <= g + 1) continue;
        if (stBoxes.length >= cap || workUsed >= WORK_CAP) return null;
        workUsed++;
        bestG.set(key, g + 1);
        const t = stBoxes.length;
        stBoxes.push(next!);
        stPlayer.push(from);
        stParent.push(s);
        stDir.push(k);
        stG.push(g + 1);
        if (solved(board.goalSet, next!)) {
          found = t;
          break;
        }
        const h = lowerBound(next!);
        if (h !== Infinity) heapPush([g + 1 + h, g + 1, t]);
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = stParent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.player;
  const boxes = board.boxes.slice();
  for (const s of chain) {
    const prevBoxes = stBoxes[stParent[s]!]!;
    const curBoxes = stBoxes[s]!;
    const from = stPlayer[s]!;
    let to = -1;
    for (const x of curBoxes) if (!prevBoxes.includes(x)) { to = x; break; }
    const k = stDir[s]!;
    const d = DIRS[k]!;
    const step = d.dy * board.w + d.dx;
    for (const x of boxes) occ[x] = 1;
    moves += walkTo(board, occ, player, from - step);
    for (const x of boxes) occ[x] = 0;
    moves += d.ch;
    player = from;
    boxes[boxes.indexOf(from)] = to;
    boxes.sort((a, b2) => a - b2);
  }
  return { pushes: chain.length, moves };
}

/** Where the player ends up after playing `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Split a level into its rooms: cut the hallway row's dead squares (the
 * player starts in the hallway, and a box there can reach no goal) out of
 * the floor, and each remaining connected part that holds boxes is a room.
 * Null when a box starts in the hallway, a goal is cut away, or a part's
 * boxes and goals do not match.
 */
function splitRooms(level: Board): { boxes: number[]; goals: number[] }[] | null {
  const live = liveSquares(level);
  const row = Math.floor(level.player / level.w);
  const cut = new Uint8Array(level.size);
  for (let i = row * level.w; i < (row + 1) * level.w; i++) {
    if (!level.wall[i] && !live[i]) cut[i] = 1;
  }
  for (const x of level.boxes) if (cut[x]) return null;
  const part = new Int32Array(level.size).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < level.size; s++) {
    if (level.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const d of DIRS) {
        const q = p + d.dy * level.w + d.dx;
        if (!level.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const x of level.boxes) {
    if (part[x] < 0) return null;
    rooms[part[x]!]!.boxes.push(x);
  }
  for (const g of level.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  for (const r of rooms) if (r.boxes.length !== r.goals.length) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one room, breadth-first first, then the greedy fallback. */
function solveRoom(board: Board): Answer {
  return solvePushBfs(board, ROOM_BFS_CAP) ?? solvePushGreedy(board, GREEDY_CAP);
}

/**
 * Solve a whole level: room by room, the other rooms' boxes standing as
 * walls (on their goals once their room is done), the player walking on
 * from where the last room left it. Falls back to solving the level whole
 * when it has no clean room split.
 */
function solveLevel(level: Board): string {
  const rooms = splitRooms(level);
  if (rooms !== null) {
    let moves = '';
    let player = level.player;
    // Each room's boxes as they stand now: on their goals once solved.
    const standing: number[][] = rooms.map((r) => r.boxes.slice());
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      const wall = level.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        for (const x of standing[oj]!) wall[x] = 1;
      }
      const subGoalSet = new Uint8Array(level.size);
      for (const g of room.goals) subGoalSet[g] = 1;
      const sub: Board = {
        w: level.w,
        h: level.h,
        size: level.size,
        wall,
        goals: room.goals.slice(),
        goalSet: subGoalSet,
        boxes: room.boxes.slice(),
        player,
      };
      const answer = solveRoom(sub);
      if (answer === null) return moves;
      moves += answer.moves;
      player = endOf(level.w, player, answer.moves);
      standing[ri] = room.goals.slice();
    }
    return moves;
  }
  const whole = solvePushBfs(level, LEVEL_BFS_CAP) ?? solvePushGreedy(level, GREEDY_CAP);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
