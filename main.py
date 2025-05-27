from ruspy_city import Game, Move

if __name__ == "__main__":
    # ruspy_city.PyGame.benchmark(10000)
    # ruspy_city.PyGame.play(7, 7)
    
    
    game = Game(7, 7)
    move = Move((0, 0), "U")
    
    Game.play_against_minimax(7, 7, 5, True)