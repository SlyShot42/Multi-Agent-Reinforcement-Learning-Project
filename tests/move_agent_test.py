from resources import MainGame, Agent, Action, GameObj
from pygame.math import Vector2

game = MainGame(2)


def test_move_agent_forward(self):
    print("\ntest_move_agent_forward\n")
    self.game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    self.game.bots[1] = Agent(Vector2(7, 5), Vector2(7, 4))
    GameObj.display_vectors(self.game.bots)
    self.game.bots[0].move_agent(Action.FORWARD)
    self.game.bots[1].move_agent(Action.FORWARD)
    GameObj.display_vectors(self.game.bots)
    self.assertEqual(self.game.bots[0], Agent(Vector2(5, 6), Vector2(5, 5)))
    self.assertEqual(self.game.bots[1], Agent(Vector2(7, 6), Vector2(7, 5)))


def test_move_agent_left(self):
    print("\ntest_move_agent_left\n")
    self.game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    self.game.bots[1] = Agent(Vector2(7, 5), Vector2(7, 4))
    GameObj.display_vectors(self.game.bots)
    self.game.bots[0].move_agent(Action.LEFT)
    self.game.bots[1].move_agent(Action.LEFT)
    GameObj.display_vectors(self.game.bots)
    self.assertEqual(self.game.bots[0], Agent(Vector2(6, 5), Vector2(5, 5)))
    self.assertEqual(self.game.bots[1], Agent(Vector2(8, 5), Vector2(7, 5)))


def test_move_agent_right(self):
    print("\ntest_move_agent_right\n")
    self.game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    self.game.bots[1] = Agent(Vector2(7, 5), Vector2(7, 4))
    GameObj.display_vectors(self.game.bots)
    self.game.bots[0].move_agent(Action.RIGHT)
    self.game.bots[1].move_agent(Action.RIGHT)
    GameObj.display_vectors(self.game.bots)
    self.assertEqual(self.game.bots[0], Agent(Vector2(4, 5), Vector2(5, 5)))
    self.assertEqual(self.game.bots[1], Agent(Vector2(6, 5), Vector2(7, 5)))


def test_move_agent_no_action(self):
    print("\ntest_move_agent_no_action\n")
    self.game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    self.game.bots[1] = Agent(Vector2(7, 5), Vector2(7, 4))
    GameObj.display_vectors(self.game.bots)
    self.game.bots[0].move_agent(Action.NO_ACTION)
    self.game.bots[1].move_agent(Action.NO_ACTION)
    GameObj.display_vectors(self.game.bots)
    self.assertEqual(self.game.bots[0], Agent(Vector2(5, 5), Vector2(5, 4)))
    self.assertEqual(self.game.bots[1], Agent(Vector2(7, 5), Vector2(7, 4)))
