use pyo3::prelude::*;
use std::cmp::PartialEq;
use std::collections::VecDeque;
use std::fmt;
use std::ops::Add;
use std::time;

use rand::Rng;

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
    fn possible_moves(&self) -> Vec<PyMove> {
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
    fn make_move(&mut self, destination: (i32, i32), place_wall: String, safe: bool) -> bool {
        let coord = Coordinate::new(destination.0, destination.1);
        let dir = match place_wall.as_str() {
            "U" => Direction::Up,
            "D" => Direction::Down,
            "L" => Direction::Left,
            "R" => Direction::Right,
            _ => return false,
        };
        let mv = Move::new(coord, dir);
        self.inner.make_move(mv, safe)
    }

    /// Check whether the game is over.
    fn game_over(&self) -> bool {
        self.inner.game_over()
    }

    /// Return (winner, (blue_score, green_score))
    fn game_result(&self) -> (String, (i32, i32)) {
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
    fn play_against_minimax(width: i32, height: i32, depth: i32, human_first: bool) {
        Game::play_against_minimax(width, height, depth, human_first);
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
            self.board_matrix.get_unchecked(coordinate.y as usize).get_unchecked(coordinate.x as usize)
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
        let x = notation.chars().nth(0).ok_or("Invalid x")?;
        let y = notation.chars().nth(1).ok_or("Invalid y")?;
        let destination = Coordinate::new(x as i32 - 'a' as i32, y as i32 - '1' as i32);
        let place_wall = match notation.chars().nth(2).unwrap() {
            'U' => Direction::Up,
            'D' => Direction::Down,
            'L' => Direction::Left,
            'R' => Direction::Right,
            _ => return Err("Invalid wall direction"),
        };

        Ok(Move::new(destination, place_wall))
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

#[derive(Debug)]
struct Score {
    blue: i32,
    green: i32,
}

#[derive(Debug)]
enum Winner {
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

    history: Vec<Move>, // history of moves
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

    fn steps_to_reach(&self, start: Coordinate) -> Board<i32> {
        let mut dist = Board::new(self.width, self.height, -1);
        let mut queue = VecDeque::new();

        // determine the "other" player so we don't move over them
        let other_player = if start == self.blue_position {
            self.green_position
        } else {
            self.blue_position
        };

        dist.set(start, 0);
        queue.push_back((start, 0));

        while !queue.is_empty() {
            let (current, d) = queue.pop_front().unwrap();
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

    fn possible_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let start = if self.blue_turn {
            self.blue_position
        } else {
            self.green_position
        };

        let reachable = self.reachable_positions(start, 3, false);

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

    fn evaluation_sorted_moves(&self) -> Vec<Move> {
        // sort the possible moves by their evaluation
        let moves = self.possible_moves();
        let mut evaluations: Vec<i32> = Vec::with_capacity(moves.len());
        let mut game_clone = self.clone();

        for mv in &moves {
            game_clone.make_move(*mv, false);
            evaluations.push(game_clone.evaluate());
            game_clone.undo_move();
        }

        let mut scored: Vec<(Move, i32)> = moves.into_iter().zip(evaluations).collect();
        if self.blue_turn {
            scored.sort_by(|a, b| b.1.cmp(&a.1)); // high to low for Blue
        } else {
            scored.sort_by(|a, b| a.1.cmp(&b.1)); // low to high for Green
        }
        scored.into_iter().map(|(mv, _)| mv).collect()
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
        // pop the last move; if there wasn't one, do nothing
        if self.history.pop().is_some() {
            // clone remaining history
            let previous = self.history.clone();
            // rebuild the game from scratch
            let w = self.width;
            let h = self.height;
            *self = Game::new(w, h);
            // replay all but the last move
            for mv in previous {
                // safe = false since these moves were already validated
                self.make_move(mv, false);
            }
        }
    }

    fn territory_difference(&self) -> i32 {
        // if it takes less steps for one player to reach a cell, the the cell is counted as the player's territory
        // always return blue territory - green territory

        let blue_dist = self.steps_to_reach(self.blue_position);
        let green_dist = self.steps_to_reach(self.green_position);

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

    fn evaluate(&self) -> i32 {
        // evaluate the game state
        // larger positive value means better for blue, larger negative value means better for green

        if self.game_over() {
            let blue_reachable =
                self.reachable_positions(self.blue_position, self.height * self.height, true);
            let green_reachable =
                self.reachable_positions(self.green_position, self.height * self.height, true);

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

    fn minimax_best_move(&self, depth: i32) -> Move {
        // before timing and scoring, order first‐level moves by a fast heuristic to improve alpha‐beta pruning

        let start = time::Instant::now();
        let mut nodes_evaluated = 0u64;
        let is_max = self.blue_turn;

        let mut game_clone = self.clone();
        // internal minimax with alpha‐beta and node counting
        fn minimax_internal(
            game: &mut Game,
            depth: i32,
            mut alpha: i32,
            mut beta: i32,
            nodes: &mut u64,
        ) -> i32 {
            *nodes += 1;
            let _moves = match depth {
                0 => return game.evaluate(),
                1 => game.possible_moves(),
                _ => game.evaluation_sorted_moves(),
            };
            let mut value = if game.blue_turn { i32::MIN } else { i32::MAX };
            for mv in _moves {
                game.make_move(mv, false);
                let score = minimax_internal(game, depth - 1, alpha, beta, nodes);
                game.undo_move();
                if game.blue_turn {
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

        // evaluate each first‐level move
        let mut scored_moves: Vec<(Move, i32)> = self
            .evaluation_sorted_moves()
            .into_iter()
            .map(|mv| {
                game_clone.make_move(mv, false);
                let score = minimax_internal(
                    &mut game_clone,
                    depth - 1,
                    i32::MIN,
                    i32::MAX,
                    &mut nodes_evaluated,
                );
                game_clone.undo_move();
                (mv, score)
            })
            .collect();

        // sort and print top 5 for diagnostics
        if is_max {
            scored_moves.sort_by(|a, b| b.1.cmp(&a.1));
        } else {
            scored_moves.sort_by(|a, b| a.1.cmp(&b.1));
        }
        println!("Top 5 moves:");
        for (mv, sc) in scored_moves.iter().take(5) {
            println!("  {:?}: {}", mv, sc);
        }
        println!("Total nodes evaluated: {}", nodes_evaluated);
        println!("Time elapsed: {:?}", start.elapsed());

        // pick randomly among all best‐scoring moves
        let best_score = scored_moves[0].1;
        let best_moves: Vec<Move> = scored_moves
            .iter()
            .filter(|&(_, sc)| *sc == best_score)
            .map(|&(mv, _)| mv)
            .collect();
        let mut rng = rand::rng();
        let best_move = best_moves[rng.random_range(0..best_moves.len())];
        best_move
    }

    fn deepening_minimax(&self, depth: i32) -> Move {
        // iterative deepening minimax
        let start = time::Instant::now();
        let best_move = self.minimax_best_move(depth);

        if start.elapsed().as_secs() < 1 {
            let best_move = self.minimax_best_move(depth + 2);
            return best_move;
        }

        best_move
    }

    fn game_over(&self) -> bool {
        //     the game is over when the green player can't reach the blue player

        let blue_reachable =
            self.reachable_positions(self.blue_position, self.height * self.height, true);

        if !blue_reachable.get(self.green_position) {
            return true;
        }
        false
    }

    fn game_result(&self) -> (Winner, Score) {
        // the score is the area of the player can reach

        let blue_reachable =
            self.reachable_positions(self.blue_position, self.height * self.height, true);
        let green_reachable =
            self.reachable_positions(self.green_position, self.height * self.height, true);

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

    fn play_against_minimax(width: i32, height: i32, depth: i32, human_first: bool) {
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
                let mv = game.deepening_minimax(depth);
                println!("AI plays {:?}", mv);
                // safe = false since minimax guarantees a legal move
                game.make_move(mv, false);
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

    pub fn minimax_self_play(width: i32, height: i32, depth: i32) {
        // play a game against itself using minimax
        let mut game = Game::new(width, height);
        loop {
            game.print();
            println!(
                "Now it's {}'s turn",
                if game.blue_turn { "Blue" } else { "Green" }
            );

            let mv = game.minimax_best_move(depth);
            println!("AI plays {:?}", mv);
            // safe = false since minimax guarantees a legal move
            game.make_move(mv, false);

            if game.game_over() {
                game.print();
                let (winner, score) = game.game_result();
                println!("Game over, {:?} wins!", winner);
                println!("Score: {} - {}", score.blue, score.green);
                break;
            }
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
        // Game::play_against_minimax(7, 7, 5, true);
        Game::minimax_self_play(7, 7, 5);
    }
}
