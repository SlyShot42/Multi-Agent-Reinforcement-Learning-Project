from itertools import combinations
from resources import MainGame

game = MainGame(2)


def collision_check():
    for i, j in combinations(range(game.num_agents), 2):
        if game.bots[i].collision(game.bots[j]):
            return True
        elif game.bots[i].collision(game.end_pts[j]):
            return True
        elif game.bots[j].collision(game.end_pts[i]):
            return True
        elif game.bots[i].collision(game.end_pts[i]):
            return True
        elif game.bots[j].collision(game.end_pts[j]):
            return True
        elif game.end_pts[i].collision(game.end_pts[j]):
            return True
        else:
            return False


def test_generate_entities():
    for i in range(100):
        game.generate_entities()
        assert collision_check() == False
