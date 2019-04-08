Play Codenames with Glove
=========================

This repository implements a simple single-player version of the codenames game
by Vlaada Chvátil.
You can play as the agent or the spymaster (not implemented yet), and the Glove 
word vectors will take the role of your partner, as you try to find the 8 marked 
words in as few rounds as possible.

The code is mostly forked from thomasahle/codenames.git. The main contribution
here is a different heuristic for finding clues.

```
$ git clone git@github.com:alireza202/codenames.git
...

$ sh get_glove.sh
...

$ python3 codenames.py
...Loading words
...Removing stopwords
...Loading vectors
...Making word to index dict
...Loading codenames
Ready!

Will you be agent or spymaster?: agent
Thinking...

**While playing, hit enter to skip**

 ladybug    white    japan      zoo     lawn
    vest  eyebrow      log   mascot     honk
    pawn      big    blimp  college     belt
 shallow    cream     lace     chew    penny
  roller      set loveseat   opener     gold


Clue: "bag 3" (remaining words 8)

Your guess: belt
Correct!
```

How it works
============
The bot decides what words go well together, by comparing their vectors in the GloVe trained on Wikipedia text.
This means that words that often occour in the same articles and sentences are judged to be similar.
In the example about, golden is of course similar to bridge by association with the Golden Gate Bridge.
Other words that were found to be similar were 'dragon', 'triangle', 'duck', 'iron' and 'horn'.

However, in Codenames the task is not merely to find words that describe other words well.
You also need to make sure that 'bad words' are as different as possible from your clue.
To achieve this, the bot tries to find a word that maximizes the similarity gap between the marked words and the bad words.

