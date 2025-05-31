from ruspy_city import Game, Move

if __name__ == "__main__":
    # ruspy_city.PyGame.benchmark(10000)
    # ruspy_city.PyGame.play(7, 7)
    
    game = Game(7, 7)
    game.print()
    game.make_move(Move.from_notation("b3L"), True)
    game.print()
    print(game.get_state_planes())