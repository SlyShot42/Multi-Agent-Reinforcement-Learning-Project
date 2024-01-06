import unittest
from main import MAIN_GAME, Agent, END_PNT, GAME_OBJ, Action
from pygame.math import Vector2

class TestAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.game = MAIN_GAME(2)

    def test_move_agent_forward(self):
        print('\ntest_move_agent_forward\n')
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(7,5),Vector2(7,4))
        GAME_OBJ.display_vectors(self.game.bots)
        self.game.bots[0].move_agent(Action.FORWARD)
        self.game.bots[1].move_agent(Action.FORWARD)
        GAME_OBJ.display_vectors(self.game.bots)
        self.assertEqual(self.game.bots[0],Agent(Vector2(5,6),Vector2(5,5)))
        self.assertEqual(self.game.bots[1],Agent(Vector2(7,6),Vector2(7,5)))

    def test_move_agent_left(self):
        print('\ntest_move_agent_left\n')
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(7,5),Vector2(7,4))
        GAME_OBJ.display_vectors(self.game.bots)
        self.game.bots[0].move_agent(Action.LEFT)
        self.game.bots[1].move_agent(Action.LEFT)
        GAME_OBJ.display_vectors(self.game.bots)
        self.assertEqual(self.game.bots[0],Agent(Vector2(6,5),Vector2(5,5)))
        self.assertEqual(self.game.bots[1],Agent(Vector2(8,5),Vector2(7,5)))

    def test_move_agent_right(self):
        print('\ntest_move_agent_right\n')
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(7,5),Vector2(7,4))
        GAME_OBJ.display_vectors(self.game.bots)
        self.game.bots[0].move_agent(Action.RIGHT)
        self.game.bots[1].move_agent(Action.RIGHT)
        GAME_OBJ.display_vectors(self.game.bots)
        self.assertEqual(self.game.bots[0],Agent(Vector2(4,5),Vector2(5,5)))
        self.assertEqual(self.game.bots[1],Agent(Vector2(6,5),Vector2(7,5)))

    def test_move_agent_no_action(self):
        print('\ntest_move_agent_no_action\n')
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(7,5),Vector2(7,4))
        GAME_OBJ.display_vectors(self.game.bots)
        self.game.bots[0].move_agent(Action.NO_ACTION)
        self.game.bots[1].move_agent(Action.NO_ACTION)
        GAME_OBJ.display_vectors(self.game.bots)
        self.assertEqual(self.game.bots[0],Agent(Vector2(5,5),Vector2(5,4)))
        self.assertEqual(self.game.bots[1],Agent(Vector2(7,5),Vector2(7,4)))

if __name__ == '__main__':
    unittest.main()