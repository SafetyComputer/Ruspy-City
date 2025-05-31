use pyo3::prelude::*;
use serde_json::json;
use std::cmp::PartialEq;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::Add;
use std::thread;
use std::time;

use rand::Rng;

#[derive(Clone)]
#[pyclass(name = "Move")]
struct PyMove {
    #[pyo3(get)]
    destination: (i32, i32),
    #[pyo3(get)]
    place_wall: String,
}

#[pymethods]
impl PyMove {
    #[new]
    fn new(destination: (i32, i32), place_wall: String) -> Self {
        PyMove {
            destination,
            place_wall,
        }
    }

    #[staticmethod]
    fn from_notation(notation: &str) -> PyResult<PyMove> {
        let mv = Move::from_notation(notation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let dest = mv.destination;
        let dir = match mv.place_wall {
            Direction::Up => "U",
            Direction::Down => "D",
            Direction::Left => "L",
            Direction::Right => "R",
        }
        .to_string();
        Ok(PyMove::new((dest.x, dest.y), dir))
    }
}

#[pyclass(name = "Game")]
struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new(width: i32, height: i32) -> Self {
        PyGame {
            inner: Game::new(width, height),
        }
    }

    /// Start an interactive game loop (blocks).
    #[staticmethod]
    fn play(width: i32, height: i32) {
        Game::play(width, height);
    }

    /// Return all legal moves as a list of PyMove.
    fn possible_moves(&mut self) -> Vec<PyMove> {
        self.inner
            .possible_moves()
            .iter()
            .map(|mv| {
                let c = mv.destination;
                let dir = match mv.place_wall {
                    Direction::Up => "U",
                    Direction::Down => "D",
                    Direction::Left => "L",
                    Direction::Right => "R",
                }
                .to_string();
                PyMove::new((c.x, c.y), dir)
            })
            .collect()
    }

    /// Attempt to make a move. `place_wall` should be "U", "D", "L" or "R".
    fn make_move(&mut self, mv: PyMove, safe: bool) -> PyResult<bool> {
        let destination = Coordinate::new(mv.destination.0, mv.destination.1);
        let place_wall = match mv.place_wall.as_str() {
            "U" => Direction::Up,
            "D" => Direction::Down,
            "L" => Direction::Left,
            "R" => Direction::Right,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid wall direction",
                ))
            }
        };
        let move_obj = Move::new(destination, place_wall);
        Ok(self.inner.make_move(move_obj, safe))
    }

    /// Check whether the game is over.
    fn game_over(&mut self) -> bool {
        self.inner.game_over()
    }

    fn get_state_planes(&self) -> Vec<Vec<Vec<bool>>> {
        let mut planes = Vec::new();

        // Blue position plane
        let mut blue_plane = Board::new(self.inner.width, self.inner.height, false);
        blue_plane.set(self.inner.blue_position, true);
        planes.push(blue_plane.board_matrix);

        // Green position plane
        let mut green_plane = Board::new(self.inner.width, self.inner.height, false);
        green_plane.set(self.inner.green_position, true);
        planes.push(green_plane.board_matrix);

        // Horizontal walls plane
        let mut horizontal_walls_plane = Board::new(self.inner.width, self.inner.height, false);
        for x in 0..self.inner.width {
            for y in 0..self.inner.height - 1 {
                horizontal_walls_plane.set(
                    Coordinate::new(x, y),
                    !self
                        .inner
                        .horizontal_walls
                        .get(Coordinate::new(x, y))
                        .is_empty(),
                );
            }
        }
        planes.push(horizontal_walls_plane.board_matrix);

        // Vertical walls plane
        let mut vertical_walls_plane = Board::new(self.inner.width, self.inner.height, false);
        for x in 0..self.inner.width - 1 {
            for y in 0..self.inner.height {
                vertical_walls_plane.set(
                    Coordinate::new(x, y),
                    !self
                        .inner
                        .vertical_walls
                        .get(Coordinate::new(x, y))
                        .is_empty(),
                );
            }
        }
        planes.push(vertical_walls_plane.board_matrix);

        let turn_indicator_plane = match self.inner.blue_turn {
            true => Board::new(self.inner.width, self.inner.height, true),
            false => Board::new(self.inner.width, self.inner.height, false),
        };
        planes.push(turn_indicator_plane.board_matrix);

        planes
    }

    /// Return (winner, (blue_score, green_score))
    fn game_result(&mut self) -> (String, (i32, i32)) {
        let (winner, score) = self.inner.game_result();
        let w = match winner {
            Winner::Blue => "Blue",
            Winner::Green => "Green",
            Winner::Draw => "Draw",
        }
        .to_string();
        (w, (score.blue, score.green))
    }

    fn print(&self) {
        self.inner.print();
    }

    /// Run a benchmark of `games` random games.
    #[staticmethod]
    fn benchmark(games: i32) {
        Game::benchmark(games)
    }

    #[staticmethod]
    fn play_against_minimax(width: i32, height: i32, human_first: bool) {
        Game::play_against_minimax(width, height, human_first);
    }
}

#[pymodule]
fn ruspy_city(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyMove>()?;
    Ok(())
}

#[derive(Clone, Copy)]
enum Cell {
    Empty,
    Blue,
    Green,
}

impl Cell {
    fn is_empty(&self) -> bool {
        match self {
            Cell::Empty => true,
            _ => false,
        }
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cell::Empty => write!(f, " "),
            Cell::Blue | Cell::Green => write!(f, "■"),
        }
    }
}
#[derive(Clone)]
struct Board<T> {
    board_matrix: Vec<Vec<T>>,
}

impl<T: Copy> Board<T> {
    fn new(width: i32, height: i32, default: T) -> Board<T> {
        Board {
            board_matrix: vec![vec![default; width as usize]; height as usize],
        }
    }

    fn get(&self, coordinate: Coordinate) -> &T {
        unsafe {
            self.board_matrix
                .get_unchecked(coordinate.y as usize)
                .get_unchecked(coordinate.x as usize)
        }
    }

    fn set(&mut self, coordinate: Coordinate, value: T) {
        self.board_matrix[coordinate.y as usize][coordinate.x as usize] = value;
    }
}

impl Board<bool> {
    fn total(&self) -> usize {
        self.board_matrix.iter().flatten().filter(|&&x| x).count()
    }
    fn clean(&mut self) {
        for i in self.board_matrix.iter_mut() {
            i.fill(false);
        }
    }
}

impl Board<i32> {
    fn clean(&mut self) {
        for i in self.board_matrix.iter_mut() {
            i.fill(-1);
        }
    }
}

impl Board<Cell> {
    fn to_bool(&self) -> Vec<Vec<bool>> {
        self.board_matrix
            .iter()
            .map(|row| row.iter().map(|&cell| !cell.is_empty()).collect())
            .collect()
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

const DIRECION_VALUES: [Direction; 4] = [
    Direction::Up,
    Direction::Down,
    Direction::Left,
    Direction::Right,
];

impl Direction {
    fn relative_position(self) -> Coordinate {
        match self {
            Direction::Up => Coordinate::new(0, -1),
            Direction::Down => Coordinate::new(0, 1),
            Direction::Left => Coordinate::new(-1, 0),
            Direction::Right => Coordinate::new(1, 0),
        }
    }
}

// implement debug for Direction
impl fmt::Debug for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Up => write!(f, "U"),
            Direction::Down => write!(f, "D"),
            Direction::Left => write!(f, "L"),
            Direction::Right => write!(f, "R"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Coordinate {
    x: i32,
    y: i32,
}

impl Coordinate {
    fn new(x: i32, y: i32) -> Coordinate {
        Coordinate { x, y }
    }

    fn inside(self, width: i32, height: i32) -> bool {
        self.x >= 0 && self.x < width && self.y >= 0 && self.y < height
    }

    fn move_to(self, direction: Direction) -> Coordinate {
        self + direction.relative_position()
    }

    fn to_tuple(self) -> (i32, i32) {
        (self.x, self.y)
    }
}

impl PartialEq for Coordinate {
    fn eq(&self, other: &Coordinate) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Add<Coordinate> for Coordinate {
    type Output = Coordinate;

    fn add(self, other: Coordinate) -> Coordinate {
        Coordinate::new(self.x + other.x, self.y + other.y)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Move {
    destination: Coordinate,
    place_wall: Direction,
}

impl Move {
    fn new(destination: Coordinate, place_wall: Direction) -> Move {
        Move {
            destination,
            place_wall,
        }
    }

    fn from_notation(notation: &str) -> Result<Move, &'static str> {
        if notation == "exit" {
            panic!("exit")
        }
        let x = notation.chars().nth(0).ok_or("Invalid Notation")?;
        let y = notation.chars().nth(1).ok_or("Invalid Notation")?;
        let destination = Coordinate::new(x as i32 - 'a' as i32, y as i32 - '1' as i32);
        let place_wall = match notation.chars().nth(2).ok_or("Invalid Notation")? {
            'U' => Direction::Up,
            'D' => Direction::Down,
            'L' => Direction::Left,
            'R' => Direction::Right,
            _ => return Err("Invalid Notation"),
        };

        Ok(Move::new(destination, place_wall))
    }

    fn to_flat(&self) -> ((i32, i32), i32) {
        // destination, wall_direction UDLR -> 0123
        let wall_direction = match self.place_wall {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        };
        (self.destination.to_tuple(), wall_direction)
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}{:?}",
            (b'a' + self.destination.x as u8) as char,
            (b'1' + self.destination.y as u8) as char,
            self.place_wall
        )
    }
}

struct EvaluatedMove {
    mv: Move,
    ev: i32,
}

impl EvaluatedMove {
    fn new(mv: Move, ev: i32) -> EvaluatedMove {
        EvaluatedMove { mv, ev }
    }
}

impl fmt::Debug for EvaluatedMove {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sign = if self.ev > 0 { "+" } else { "" };
        write!(f, "{:?} ({}{})", self.mv, sign, self.ev)
    }
}

impl PartialEq for EvaluatedMove {
    // compare only by evaluation value
    fn eq(&self, other: &Self) -> bool {
        self.ev == other.ev
    }
}

impl Eq for EvaluatedMove {}

impl PartialOrd for EvaluatedMove {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.ev.cmp(&other.ev))
    }
}

impl Ord for EvaluatedMove {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ev.cmp(&other.ev)
    }
}

#[derive(Debug)]
struct Score {
    blue: i32,
    green: i32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Winner {
    Blue,
    Green,
    Draw,
}

#[derive(Clone)]
pub struct Game {
    width: i32,
    height: i32,

    blue_position: Coordinate,
    green_position: Coordinate,

    horizontal_walls: Board<Cell>, // 0 for no wall, 1 for blue wall, 2 for green wall
    vertical_walls: Board<Cell>,

    blue_turn: bool, // true for blue, false for green

    history: Vec<Move>,                    // history of moves
    evaluation_cache: BTreeMap<u128, i32>, // cache for evaluation
    blue_reachable_cache: Board<bool>,
    green_reachable_cache: Board<bool>,
    blue_steps_cache: Board<i32>,
    green_steps_cache: Board<i32>,
}

impl Game {
    fn new(width: i32, height: i32) -> Game {
        Game {
            width,
            height,
            blue_position: Coordinate::new(0, 0), // the top-left corner
            green_position: Coordinate::new(width - 1, height - 1), // bottom-right corner
            horizontal_walls: Board::new(width, height - 1, Cell::Empty),
            vertical_walls: Board::new(width - 1, height, Cell::Empty),
            blue_turn: true,
            history: Vec::new(),
            evaluation_cache: BTreeMap::new(),
            blue_reachable_cache: Board::new(width, height, false),
            green_reachable_cache: Board::new(width, height, false),
            blue_steps_cache: Board::new(width, height, -1),
            green_steps_cache: Board::new(width, height, -1),
        }
    }

    fn reachable_with_cache(&mut self, start: Coordinate, step: i32, ignore_other_player: bool) {
        let reachable = if start == self.blue_position {
            self.blue_reachable_cache.clean();
            &mut self.blue_reachable_cache
        } else {
            self.green_reachable_cache.clean();
            &mut self.green_reachable_cache
        };
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        reachable.set(start, true);

        // determine the "other" player so we don't move over them
        let other_player = if start == self.blue_position {
            self.green_position
        } else {
            self.blue_position
        };

        while !queue.is_empty() {
            let (current, current_step) = queue.pop_front().unwrap();
            if current_step == step {
                continue;
            }

            for direction in DIRECION_VALUES {
                let next = current.move_to(direction);
                if !next.inside(self.width, self.height) {
                    continue;
                }
                // don't move over the other player's pawn
                if next == other_player && !ignore_other_player {
                    continue;
                }
                if *reachable.get(next) {
                    continue;
                }
                // check walls
                if (direction == Direction::Right && self.vertical_walls.get(current).is_empty())
                    || (direction == Direction::Left && self.vertical_walls.get(next).is_empty())
                    || (direction == Direction::Down
                        && self.horizontal_walls.get(current).is_empty())
                    || (direction == Direction::Up && self.horizontal_walls.get(next).is_empty())
                {
                    reachable.set(next, true);
                    queue.push_back((next, current_step + 1));
                }
            }
        }
    }

    fn reachable_positions(
        &self,
        start: Coordinate,
        step: i32,
        ignore_other_player: bool,
    ) -> Board<bool> {
        let mut reachable = Board::new(self.width, self.height, false);
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        reachable.set(start, true);

        // determine the "other" player so we don't move over them
        let other_player = if start == self.blue_position {
            self.green_position
        } else {
            self.blue_position
        };

        while !queue.is_empty() {
            let (current, current_step) = queue.pop_front().unwrap();
            if current_step == step {
                continue;
            }

            for direction in DIRECION_VALUES {
                let next = current.move_to(direction);
                if !next.inside(self.width, self.height) {
                    continue;
                }
                // don't move over the other player's pawn
                if next == other_player && !ignore_other_player {
                    continue;
                }
                if *reachable.get(next) {
                    continue;
                }
                // check walls
                if (direction == Direction::Right && self.vertical_walls.get(current).is_empty())
                    || (direction == Direction::Left && self.vertical_walls.get(next).is_empty())
                    || (direction == Direction::Down
                        && self.horizontal_walls.get(current).is_empty())
                    || (direction == Direction::Up && self.horizontal_walls.get(next).is_empty())
                {
                    reachable.set(next, true);
                    queue.push_back((next, current_step + 1));
                }
            }
        }

        reachable
    }
    fn steps_with_cache(&mut self, start: Coordinate) {
        let dist = if start == self.blue_position {
            self.blue_steps_cache.clean();
            &mut self.blue_steps_cache
        } else {
            self.green_steps_cache.clean();
            &mut self.green_steps_cache
        };
        let mut queue: VecDeque<(Coordinate, i32)> = VecDeque::new();

        // determine the "other" player so we don't move over them
        let other_player = if start == self.blue_position {
            self.green_position
        } else {
            self.blue_position
        };

        dist.set(start, 0);
        queue.push_back((start, 0));

        while !queue.is_empty() {
            let (current, d) = queue.remove(0).unwrap();
            for dir in DIRECION_VALUES {
                let next = current.move_to(dir);
                if !next.inside(self.width, self.height) {
                    continue;
                }
                // don't move over the other player's pawn
                if next == other_player {
                    continue;
                }
                // already visited?
                if *dist.get(next) != -1 {
                    continue;
                }
                // check walls
                let can_move = match dir {
                    Direction::Right => self.vertical_walls.get(current).is_empty(),
                    Direction::Left => self.vertical_walls.get(next).is_empty(),
                    Direction::Down => self.horizontal_walls.get(current).is_empty(),
                    Direction::Up => self.horizontal_walls.get(next).is_empty(),
                };
                if can_move {
                    dist.set(next, d + 1);
                    queue.push_back((next, d + 1));
                }
            }
        }
    }
    fn steps_to_reach(&self, start: Coordinate) -> Board<i32> {
        let mut dist = Board::new(self.width, self.height, -1);
        let mut queue: VecDeque<(Coordinate, i32)> = VecDeque::new();

        // determine the "other" player so we don't move over them
        let other_player = if start == self.blue_position {
            self.green_position
        } else {
            self.blue_position
        };

        dist.set(start, 0);
        queue.push_back((start, 0));

        while !queue.is_empty() {
            let (current, d) = queue.remove(0).unwrap();
            for dir in DIRECION_VALUES {
                let next = current.move_to(dir);
                if !next.inside(self.width, self.height) {
                    continue;
                }
                // don't move over the other player's pawn
                if next == other_player {
                    continue;
                }
                // already visited?
                if *dist.get(next) != -1 {
                    continue;
                }
                // check walls
                let can_move = match dir {
                    Direction::Right => self.vertical_walls.get(current).is_empty(),
                    Direction::Left => self.vertical_walls.get(next).is_empty(),
                    Direction::Down => self.horizontal_walls.get(current).is_empty(),
                    Direction::Up => self.horizontal_walls.get(next).is_empty(),
                };
                if can_move {
                    dist.set(next, d + 1);
                    queue.push_back((next, d + 1));
                }
            }
        }

        dist
    }

    fn possible_moves(&mut self) -> Vec<Move> {
        let mut moves = Vec::new();
        let start = if self.blue_turn {
            self.blue_position
        } else {
            self.green_position
        };
        self.reachable_with_cache(start, 3, false);
        let reachable = if self.blue_turn {
            &mut self.blue_reachable_cache
        } else {
            &mut self.green_reachable_cache
        };

        // get all possible moves
        // the wall placement is possible if the wall is not placed and the edge is not on the border

        for y in 0..self.height {
            for x in 0..self.width {
                if !reachable.get(Coordinate::new(x, y)) {
                    continue;
                }

                let current = Coordinate::new(x, y);
                for direction in DIRECION_VALUES {
                    let next = current.move_to(direction);
                    if !next.inside(self.width, self.height) {
                        continue;
                    }

                    if direction == Direction::Right && self.vertical_walls.get(current).is_empty()
                        || direction == Direction::Left && self.vertical_walls.get(next).is_empty()
                        || direction == Direction::Down
                            && self.horizontal_walls.get(current).is_empty()
                        || direction == Direction::Up && self.horizontal_walls.get(next).is_empty()
                    {
                        moves.push(Move {
                            destination: current,
                            place_wall: direction,
                        });
                    }
                }
            }
        }

        moves
    }

    fn evaluation_sorted_moves(&mut self, cutoff: i32) -> Vec<Move> {
        // evaluate all possible moves and return them sorted by evaluation value
        let mut scored_moves: Vec<EvaluatedMove> = self
            .possible_moves()
            .into_iter()
            .map(|mv| {
                let score = self.evaluate_move(mv);
                EvaluatedMove::new(mv, score)
            })
            .collect();

        // sort moves by evaluation value
        scored_moves.sort();
        if self.blue_turn {
            scored_moves.reverse(); // descending for max player
        }

        if cutoff > 0 && scored_moves.len() > cutoff as usize {
            scored_moves.truncate(cutoff as usize);
        }

        scored_moves.into_iter().map(|em| em.mv).collect()
    }

    fn make_move(&mut self, mv: Move, safe: bool) -> bool {
        // make the move
        // if mv is in possible_moves, then make the move and place

        // return true if the move is made, false otherwise

        if safe && !self.possible_moves().contains(&mv) {
            return false;
        }

        if self.blue_turn {
            self.blue_position = mv.destination;
        } else {
            self.green_position = mv.destination;
        }

        let cell = if self.blue_turn {
            Cell::Blue
        } else {
            Cell::Green
        };

        match mv.place_wall {
            Direction::Up => self
                .horizontal_walls
                .set(mv.destination.move_to(Direction::Up), cell),
            Direction::Down => self.horizontal_walls.set(mv.destination, cell),
            Direction::Left => self
                .vertical_walls
                .set(mv.destination.move_to(Direction::Left), cell),
            Direction::Right => self.vertical_walls.set(mv.destination, cell),
        }

        self.blue_turn = !self.blue_turn;
        self.history.push(mv);
        true
    }

    fn undo_move(&mut self) {
        let last_move = self.history.pop().expect("No moves to undo");

        let last_position = match self.history.len() {
            0 => Coordinate { x: 0, y: 0 }, // if no moves left, reset to start
            1 => Coordinate {
                x: self.width - 1,
                y: self.height - 1,
            }, // if only one move left, reset to green position
            _ => {
                let second_last_move = self.history[self.history.len() - 2];
                second_last_move.destination
            }
        };

        // remove the wall
        match last_move.place_wall {
            Direction::Up => self
                .horizontal_walls
                .set(last_move.destination.move_to(Direction::Up), Cell::Empty),
            Direction::Down => self
                .horizontal_walls
                .set(last_move.destination, Cell::Empty),
            Direction::Left => self
                .vertical_walls
                .set(last_move.destination.move_to(Direction::Left), Cell::Empty),
            Direction::Right => self.vertical_walls.set(last_move.destination, Cell::Empty),
        }

        // reset the position
        if self.blue_turn {
            self.green_position = last_position;
        } else {
            self.blue_position = last_position;
        }

        self.blue_turn = !self.blue_turn;
    }

    fn territory_difference(&mut self) -> i32 {
        // if it takes less steps for one player to reach a cell, the the cell is counted as the player's territory
        // always return blue territory - green territory
        self.steps_with_cache(self.blue_position);
        self.steps_with_cache(self.green_position);
        let blue_dist = &mut self.blue_steps_cache;
        let green_dist = &mut self.green_steps_cache;

        let mut blue_territory = 0;
        let mut green_territory = 0;

        for y in 0..self.height {
            for x in 0..self.width {
                let pos = Coordinate::new(x, y);
                let bd = *blue_dist.get(pos);
                let gd = *green_dist.get(pos);

                // if blue reaches faster (or green can't reach), count for blue
                if bd >= 0 && (gd < 0 || bd < gd) {
                    blue_territory += 1;
                }
                // if green reaches faster (or blue can't reach), count for green
                else if gd >= 0 && (bd < 0 || gd < bd) {
                    green_territory += 1;
                }
            }
        }

        blue_territory - green_territory
    }

    fn evaluate(&mut self) -> i32 {
        // evaluate the game state
        // larger positive value means better for blue, larger negative value means better for green

        if self.game_over() {
            self.reachable_with_cache(self.blue_position, self.height * self.height, true);
            self.reachable_with_cache(self.green_position, self.height * self.height, true);
            let blue_reachable = &mut self.blue_reachable_cache;
            let green_reachable = &mut self.green_reachable_cache;

            let blue_score = blue_reachable.total() as i32;
            let green_score = green_reachable.total() as i32;

            if blue_score > green_score {
                return 1000; // Blue wins
            } else if green_score > blue_score {
                return -1000; // Green wins
            } else {
                return 0; // Draw
            }
        }

        let territory_diff = self.territory_difference();
        territory_diff
    }

    fn evaluate_with_cache(&mut self) -> (i32, bool) {
        // if cached
        let current_position = self.convert_current_position();
        match self.evaluation_cache.get(&current_position) {
            Some(&score) => return (score, true), // return cached score and true for cache hit
            None => {
                // evaluate and cache the result
                let score = self.evaluate();
                self.evaluation_cache.insert(current_position, score);
                return (score, false); // return score and false for cache miss
            }
        }
    }

    fn convert_current_position(&self) -> u128 {
        // convert the current position to a unique u128 value
        let mut value: u128 = 0;
        value |= (self.blue_position.x as u128) << 112;
        value |= (self.blue_position.y as u128) << 104;
        value |= (self.green_position.x as u128) << 96;
        value |= (self.green_position.y as u128) << 88;

        for x in 0..self.width {
            for y in 0..self.height - 1 {
                let cell = self.horizontal_walls.get(Coordinate::new(x, y));
                if !cell.is_empty() {
                    value |= (1u128 << (x + y * self.width)) << 42;
                }
            }
        }

        for x in 0..self.width - 1 {
            for y in 0..self.height {
                let cell = self.vertical_walls.get(Coordinate::new(x, y));
                if !cell.is_empty() {
                    value |= (1u128 << (x + y * self.width)) << 0;
                }
            }
        }

        value
    }

    fn minimax_evaluate(
        &mut self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        nodes: &mut u64,
        cutoff: i32,
    ) -> i32 {
        *nodes += 1;
        let moves = match depth {
            0 => return self.evaluate(),
            1 => self.possible_moves(),
            _ => self.evaluation_sorted_moves(cutoff),
        };
        let mut value = if self.blue_turn { i32::MIN } else { i32::MAX };
        for mv in moves {
            self.make_move(mv, false);
            let score = self.minimax_evaluate(depth - 1, alpha, beta, nodes, cutoff);
            self.undo_move();
            if self.blue_turn {
                value = value.max(score);
                alpha = alpha.max(value);
            } else {
                value = value.min(score);
                beta = beta.min(value);
            }
            if alpha >= beta {
                break;
            }
        }
        value
    }

    fn minimax_best_move(&mut self, depth: i32, cutoff: i32) -> Move {
        let mut node_evaluated = 0u64;
        let start = time::Instant::now();
        let evaluated_moves = self.minimax_evaluate_moves(depth, &mut node_evaluated);
        // diagnostics
        println!("Top 5 moves:");
        for em in evaluated_moves.iter().take(5) {
            println!("  {:?}", em);
        }
        println!("Total nodes evaluated: {}", node_evaluated);
        println!("Time elapsed: {:?}", start.elapsed());

        // pick randomly among best‐scoring
        let best_score = evaluated_moves[0].ev;
        let best_moves: Vec<Move> = evaluated_moves
            .iter()
            .filter(|em| em.ev == best_score)
            .map(|em| em.mv)
            .collect();
        let mut rng = rand::rng();
        best_moves[rng.random_range(0..best_moves.len())]
    }

    fn minimax_evaluate_moves(&mut self, depth: i32, nodes: &mut u64) -> Vec<EvaluatedMove> {
        // evaluate all first‐level moves and return them sorted
        let mut scored: Vec<EvaluatedMove> = self
            .evaluation_sorted_moves(0)
            .into_iter()
            .map(|mv| {
                self.make_move(mv, false);
                let sc = self.minimax_evaluate(depth - 1, i32::MIN, i32::MAX, nodes, 0);
                self.undo_move();
                EvaluatedMove::new(mv, sc)
            })
            .collect();

        scored.sort();

        if self.blue_turn {
            scored.reverse(); // descending for max player
        }

        scored
    }

    fn iterative_deepening_minimax(&mut self, depth: i32) -> EvaluatedMove {
        // iterative deepening minimax with aspiration windows
        let start = time::Instant::now();
        let max_depth = match self.history.len().cmp(&6) {
            std::cmp::Ordering::Less => depth + 2, // if less than 6 moves, use 6
            _ => depth + 4,                        // otherwise, use history length + 2
        }; // Maximum depth to search
        let time_limit_secs = 3; // Time limit in seconds

        // Initial search at the base depth to get a starting value
        let evaluated_moves = self.minimax_evaluate_moves(depth, &mut 0u64);
        let mut best_move = evaluated_moves[0].mv;
        let mut best_score = evaluated_moves[0].ev;
        let mut current_depth = depth + 2;

        // Window size parameters
        let mut window_size = 1; // Initial window size

        // Main iterative deepening loop
        while current_depth <= max_depth && start.elapsed().as_secs() < time_limit_secs {
            println!(
                "Searching at depth {} with window around {}",
                current_depth, best_score
            );

            // Set aspiration window bounds
            let mut alpha = best_score - window_size;
            let mut beta = best_score + window_size;
            let mut retry = true;

            // Try search with current window, expand if needed
            while retry {
                retry = false;
                let mut nodes_evaluated = 0u64;

                // Score each first-level move with the current window
                let mut scored: Vec<EvaluatedMove> = self
                    .evaluation_sorted_moves(0)
                    .into_iter()
                    .map(|mv| {
                        self.make_move(mv, false);
                        let sc = self.minimax_evaluate(
                            current_depth - 1,
                            alpha,
                            beta,
                            &mut nodes_evaluated,
                            0,
                        );
                        self.undo_move();
                        EvaluatedMove::new(mv, sc)
                    })
                    .collect();

                // If score is outside window bounds, retry with wider window
                if !scored.is_empty() {
                    scored.sort();
                    if self.blue_turn {
                        scored.reverse();
                    }

                    let new_score = scored[0].ev;

                    // Check if result was outside the window
                    if new_score <= alpha {
                        // Failed low, retry with wider window
                        println!("Failed low: {} <= {}, widening window", new_score, alpha);
                        window_size *= 2;
                        alpha = new_score - window_size;
                        retry = true;
                        continue;
                    } else if new_score >= beta {
                        // Failed high, retry with wider window
                        println!("Failed high: {} >= {}, widening window", new_score, beta);
                        window_size *= 2;
                        beta = new_score + window_size;
                        retry = true;
                        continue;
                    } else {
                        // Search succeeded within window
                        best_score = new_score;

                        // Pick randomly among best-scoring moves
                        let best_moves: Vec<Move> = scored
                            .iter()
                            .filter(|em| em.ev == new_score)
                            .map(|em| em.mv)
                            .collect();

                        let mut rng = rand::rng();
                        best_move = best_moves[rng.random_range(0..best_moves.len())];

                        // Diagnostics
                        println!("Top 5 moves:");
                        for em in scored.iter().take(5) {
                            println!("  {:?}", em);
                        }
                        println!("Nodes evaluated: {}", nodes_evaluated);
                        println!("Elapsed time: {:?}", start.elapsed());
                    }
                }
            }

            // Reset window size for next iteration
            window_size = 1;

            // Increase depth for next iteration
            current_depth += 2;
        }

        println!("Final best move: {:?} with score {}", best_move, best_score);
        EvaluatedMove::new(best_move, best_score)
    }

    // Helper function to evaluate a specific move
    fn evaluate_move(&mut self, mv: Move) -> i32 {
        self.make_move(mv, false);
        let score = self.evaluate();
        self.undo_move();
        score
    }

    fn game_over(&mut self) -> bool {
        //     the game is over when the green player can't reach the blue player
        self.reachable_with_cache(self.blue_position, self.height * self.height, true);
        let blue_reachable = &mut self.blue_reachable_cache;

        if !blue_reachable.get(self.green_position) {
            return true;
        }
        false
    }

    fn game_result(&mut self) -> (Winner, Score) {
        // the score is the area of the player can reach
        self.reachable_with_cache(self.blue_position, self.height * self.height, true);
        self.reachable_with_cache(self.green_position, self.height * self.height, true);
        let blue_reachable = &mut self.blue_reachable_cache;
        let green_reachable = &mut self.green_reachable_cache;

        let blue_score = blue_reachable.total() as i32;
        let green_score = green_reachable.total() as i32;

        let score = Score {
            blue: blue_score,
            green: green_score,
        };

        match blue_score.cmp(&green_score) {
            std::cmp::Ordering::Greater => (Winner::Blue, score),
            std::cmp::Ordering::Less => (Winner::Green, score),
            std::cmp::Ordering::Equal => (Winner::Draw, score),
        }
    }

    fn play(width: i32, height: i32) {
        let mut game = Game::new(width, height);

        loop {
            println!("===============================");
            println!("     Welcome to Rusty City! ");
            println!("===============================");

            game.print();
            println!(
                "Now it's {}'s turn",
                if game.blue_turn { "Blue" } else { "Green" }
            );

            loop {
                println!("Enter your move: ");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();

                if input.trim() == "exit" || input.trim() == "quit" {
                    println!("Exiting the game.");
                    return;
                }

                if input.trim() == "undo" {
                    game.undo_move();
                    break;
                }

                let mv = Move::from_notation(input.trim());
                if mv.is_err() {
                    println!("Invalid move format, try again.");
                    continue;
                }
                let mv = mv.unwrap();
                let res = game.make_move(mv, true);
                if res {
                    break;
                }
                println!("Invalid move");
            }

            if game.game_over() {
                game.print();
                let (winner, score) = game.game_result();
                println!("Game over, {:?} wins!", winner);
                println!("{0} - {1}", score.blue, score.green);

                break;
            }
        }
    }

    fn play_against_minimax(width: i32, height: i32, human_first: bool) {
        let mut game = Game::new(width, height);

        loop {
            game.print();
            println!(
                "Now it's {}'s turn",
                if game.blue_turn { "Blue" } else { "Green" }
            );

            // if it's the human's turn
            if game.blue_turn == human_first {
                // human move
                loop {
                    println!("Enter your move: ");
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).unwrap();
                    let mv = Move::from_notation(input.trim());
                    if mv.is_err() {
                        println!("Invalid move format, try again.");
                        continue;
                    }
                    let mv = mv.unwrap();
                    let res = game.make_move(mv, true);
                    if res {
                        break;
                    }
                    println!("Invalid move, try again.");
                }
            } else {
                // AI move
                println!("AI thinking...");
                let ev_mv = game.iterative_deepening_minimax(2);
                println!("AI plays {:?}", ev_mv.mv);
                // safe = false since minimax guarantees a legal move
                game.make_move(ev_mv.mv, false);
            }

            if game.game_over() {
                game.print();
                let (winner, score) = game.game_result();
                println!("Game over, {:?} wins!", winner);
                println!("Score: {} - {}", score.blue, score.green);
                break;
            }
        }
    }

    fn minimax_self_play(width: i32, height: i32, record_moves: bool, file_path: &str) -> Winner {
        // play a game against itself using minimax
        let mut game = Game::new(width, height);
        let mut json_buffer: Vec<serde_json::Value> = Vec::new();
        let winner: Winner;

        loop {
            game.print();
            println!(
                "Now it's {}'s turn",
                if game.blue_turn { "Blue" } else { "Green" }
            );

            let eval_mv = match game.blue_turn {
                true => game.iterative_deepening_minimax(1),
                false => game.iterative_deepening_minimax(1),
            };
            println!("AI plays {:?}", eval_mv.mv);
            // safe = false since minimax guarantees a legal move

            if record_moves {
                let mv = eval_mv.mv;
                let ev = eval_mv.ev;

                let json = json!({
                    "hor_wall": game.horizontal_walls.to_bool(),
                    "ver_wall": game.vertical_walls.to_bool(),
                    "blue_pos": game.blue_position.to_tuple(),
                    "green_pos": game.green_position.to_tuple(),
                    "blue_turn": game.blue_turn,
                    "best_move": mv.to_flat(),
                    "evaluation": ev,
                });

                json_buffer.push(json);
            }
            game.make_move(eval_mv.mv, false);

            if game.game_over() {
                game.print();
                let (_winner, score) = game.game_result();
                winner = _winner;
                println!("Game over, {:?} wins!", winner);
                println!("Score: {} - {}", score.blue, score.green);
                break;
            }
        }
        // append to file
        if record_moves {
            {
                // Ensure the directory "log" exists before running
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file_path)
                    .expect("Failed to open or create log/self_play.json");
                for entry in &json_buffer {
                    let line =
                        serde_json::to_string(entry).expect("Failed to serialize JSON entry");
                    writeln!(file, "{}", line).expect("Failed to write to log/self_play.json");
                }
            }
        }

        winner
    }

    fn multithread_self_play(width: i32, height: i32, num_threads: i32) {
        let mut handles = Vec::with_capacity(num_threads as usize);

        for i in 0..num_threads {
            let file_path = format!("log/self_play_thread_{}.json", i);
            let width = width;
            let height = height;

            let handle = thread::spawn(move || {
                // run one self‐play in this thread, recording moves to its own file
                Game::minimax_self_play(width, height, true, &file_path);
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }
    }

    fn print(&self) {
        // print the cells and walls
        // use table characters

        print!("  ┌");
        for x in 0..self.height {
            if x != 0 {
                print!("┬");
            }
            print!("───");
        }
        println!("┐");

        for y in 0..self.width {
            print!("{} │ ", y + 1);

            for x in 0..self.height {
                let cell_coordinate = Coordinate::new(x, y);

                if self.blue_position == cell_coordinate {
                    print!("B");
                } else if self.green_position == cell_coordinate {
                    print!("G");
                } else {
                    print!(" ");
                }

                if x < self.height - 1 {
                    if self.vertical_walls.get(cell_coordinate).is_empty() {
                        print!("   ");
                    } else {
                        print!(" ┃ ");
                    }
                }
            }

            print!(" │");
            println!();

            if y < self.width - 1 {
                print!("  ├");
                for x in 0..self.height {
                    let cell_coordinate = Coordinate::new(x, y);

                    if x != 0 {
                        print!("┼");
                    }
                    if self.horizontal_walls.get(cell_coordinate).is_empty() == true {
                        print!("   ");
                    } else {
                        print!("━━━");
                    }
                }
                println!("┤");
            }
        }

        print!("  └");
        for x in 0..self.height {
            if x != 0 {
                print!("┴");
            }
            print!("───");
        }
        println!("┘");

        print!("    ");
        for x in 0..self.height {
            print!("{}   ", (b'a' + x as u8) as char);
        }
        println!();
    }

    pub fn benchmark(games: i32) {
        let mut moves = 0;
        let time = time::Instant::now();

        for _ in 0..games {
            let mut game = Game::new(7, 7);
            loop {
                let possible_moves = game.possible_moves();

                // randomly select a move
                let mv =
                    possible_moves[Rng::random_range(&mut rand::rng(), 0..possible_moves.len())];
                game.make_move(mv, false);
                moves += 1;

                if game.game_over() {
                    break;
                }
            }
        }
        println!(
            "Played {} games with {} moves in {:?}",
            games,
            moves,
            time.elapsed()
        );
        println!("Average time per game: {:?}", time.elapsed() / games as u32);
        println!("Average time per move: {:?}", time.elapsed() / moves);
    }
}

#[cfg(test)]
mod tests {
    use crate::Game;
    use crate::Move;
    use crate::Winner;
    use std::time::Instant;

    // #[test]
    // fn benchmark() {
    //     // evaluate(): 178172/s (5.61s for 1M calls)
    //     // game_over(): 546218/s (1.83s for 1M calls)
    //     // possible_moves(): 1074137/s (0.931s for 1M calls)

    //     let game = Game::new(7, 7);
    //     let n: i32 = 1000000;
    //     let start = Instant::now();
    //     for _ in 0..n {
    //         game.possible_moves();
    //     }
    //     let elapsed = start.elapsed();
    //     println!(
    //         "Benchmarking {} calls took {:?} ({:2} calls per second)",
    //         n,
    //         elapsed,
    //         n as f64 / elapsed.as_secs_f64()
    //     );
    // }

    // #[test]
    // fn bench() {
    //     Game::benchmark(10000);
    // }

    #[test]
    fn play() {
        // Game::play_against_minimax(7, 7, true);
        // Game::play(7, 7);
        loop {
            Game::multithread_self_play(7, 7, 8);
        }
    }
}
