from resources import MainGame, Action
from resources.config import CELL_NUMBER
from pygame.math import Vector2
import numpy as np

game = MainGame(2)


def move_all_to_positions(game):
    for i in range(game.num_agents):
        game.move_to_positions(
            game.bots[i].front, game.bots[i].back, game.end_pts[i].pos
        )


def move_all_to_extractions(game):
    for i in range(game.num_agents):
        game.move_to_extractions(
            game.bots[i].front, game.bots[i].back, game.end_pts[i].pos
        )


@pytest.mark.first
def test_equal_score_gain():
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(5, 7), Vector2(4, 7))
    game.end_pts[0].place(Vector2(8, 5))
    game.end_pts[1].place(Vector2(8, 7))
    move_all_to_extractions(game)
    distance = game.end_pts[0].pos - game.bots[0].front
    rewards, dones, scores = 0, 0, 0
    for i in range(distance.x):
        rewards, dones, scores = game.step([Action.FORWARD, Action.FORWARD])
    assert rewards == np.array([10, 10])
    assert dones == np.zeros(game.num_agents)
    assert scores == np.ones(game.num_agents)
    game.move_to_positions(game.end_pts[0].pos, game.end_pts[1].pos)
    game.end_pts[0].place(Vector2(8, 3))
    game.end_pts[1].place(Vector2(8, 9))
    game.move_to_extractions(game.end_pts[0].pos, game.end_pts[1].pos)
    distance = game.end_pts[1].pos - game.bots[1].front
    game.step([Action.LEFT, Action.RIGHT])
    for i in range(distance.y - 1):
        rewards, dones, scores = game.step([Action.FORWARD, Action.FORWARD])
    assert rewards == np.array([10, 10])
    assert dones == np.zeros(game.num_agents)
    assert scores == np.array([2, 2])


@pytest.mark.second
def test_unequal_score_gain():
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(5, 7), Vector2(4, 7))
    game.end_pts[0].place(Vector2(8, 5))
    game.end_pts[1].place(Vector2(8, 7))
    move_all_to_extractions(game)
    distance = game.end_pts[0].pos - game.bots[0].front
    rewards, dones, scores = 0, 0, 0
    for i in range(distance.x):
        rewards, dones, scores = game.step([Action.FORWARD, Action.NO_ACTION])
    assert rewards == np.array([10, 0])
    assert dones == np.zeros(game.num_agents)
    assert scores == np.array([3, 2])


@pytest.mark.third
def test_bot_to_bot_collision():
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(5, 6), Vector2(5, 7))
    game.end_pts[0].place(Vector2(8, 5))
    game.end_pts[1].place(Vector2(8, 7))
    move_all_to_extractions(game)
    rewards, dones, scores = game.step([Action.FORWARD, Action.FORWARD])
    assert rewards == np.array([-10, -10])
    assert dones == np.ones(game.num_agents)
    assert scores == np.zeros(game.num_agents)


def test_unequal_score_collision():
    game = MainGame(3)
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(5, 7), Vector2(4, 7))
    game.bots[2].place(Vector2(5, 8), Vector2(5, 9))
    game.end_pts[0].place(Vector2(6, 5))
    game.end_pts[1].place(Vector2(8, 7))
    game.end_pts[2].place(Vector2(8, 6))
    move_all_to_extractions(game)
    rewards, dones, scores = game.step([Action.FORWARD, Action.FORWARD, Action.FORWARD])
    assert rewards == np.array([10, -10, -10])
    assert dones == np.array([0, 1, 1])
    assert scores == np.array([1, 0, 0])


def test_death_boundary():
    game = MainGame(1)
    move_all_to_positions(game)
    game.bots[0].place(Vector2(CELL_NUMBER, 5), Vector2(CELL_NUMBER - 1, 5))
    game.end_pts[0].place(Vector2(0, 5))
    move_all_to_extractions(game)
    rewards, dones, scores = game.step([Action.FORWARD])
    assert rewards == np.array([-10])
    assert dones == np.ones(game.num_agents)
    assert scores == np.zeros(game.num_agents)
