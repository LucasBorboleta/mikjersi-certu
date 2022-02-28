# Mikjersi certu

The Python package *mikjersi_certu* provides a GUI and a rules engine for playing the *Mikjersi* board-game, which is a micro-variant of [*Jersi*](https://github.com/LucasBorboleta/jersi) board-game. 

<img src="./docs/gui-game-play-minimax-2-vs-minimax-1.gif" style="zoom:75%;" />

The rules of Mikjersi are those of Jersi, but with the following major changes (https://github.com/LucasBorboleta/mikjersi):

- **Board** -- The board is made of 5x5 squared cells.

- **Moves** -- The moves are only vertically and horizontally, but not along diagonal.
- **Cubes** -- Each player owns 7 cubes:  rock, paper, scissors, fool, king, mountain and wise.
- **Exchange of prisoners** -- For 2 turns during a game, when a player captures cubes whose sorts exist as prisoners by his opponent, then those prisoners are immediately exchanged and moved into their respective reserves.

Above is an overview of the GUI interface. This is a work in progress.

*mikjersi_certu* is being developed on Windows and should be portable on Linux. For running it on your computer read the [**INSTALL**](./docs/INSTALL.md) instructions.

If you intent to derive or to sell either a text, a product or a software from this work, then read the [**LICENSE**](./docs/LICENSE.txt) and the  [**COPYRIGHT**](./docs/COPYRIGHT.md)  documents.

