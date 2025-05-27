from ruspy_city import PyGame, PyMove

if __name__ == "__main__":
    # ruspy_city.PyGame.benchmark(10000)
    # ruspy_city.PyGame.play(7, 7)
    
    
    game = PyGame(7, 7)
    move = PyMove((0, 0), "U")
    
    game.print()
    print(game.possible_moves())