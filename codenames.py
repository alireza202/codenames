import sys
import random
import re
import numpy as np
import math

from typing import List, Tuple, Iterable
from nltk.metrics.distance import edit_distance

SIMILARTY_THRESHOLD = 0.3

# This file stores the "solutions" the bot had intended,
# when you play as agent and the bot as spymaster.
log_file = open("log_file", "w")

def similar_words(w1, w2):

    if (w1 + 'ing' == w2) or (w2 + 'ing' == w1):
        return True
    else:
        return False


class Reader:
    def read_picks(
        self, words: List[str], my_words: Iterable[str], cnt: int
    ) -> List[str]:
        """
        Query the user for guesses.
        :param words: Words the user can choose from.
        :param my_words: Correct words.
        :param cnt: Number of guesses the user has.
        :return: The words picked by the user.
        """
        raise NotImplementedError

    def read_clue(self, word_set: Iterable[str]) -> Tuple[str, int]:
        """
        Read a clue from the (spymaster) user.
        :param word_set: Valid words
        :return: The clue and number given.
        """
        raise NotImplementedError

    def print_words(self, words: List[str], nrows: int):
        """
        Prints a list of words as a 2d table, using `nrows` rows.
        :param words: Words to be printed.
        :param nrows: Number of rows to print.
        """
        raise NotImplementedError


class TerminalReader(Reader):
    def read_picks(
        self, 
        words: List[str], 
        my_words: Iterable[str], 
        cnt: int
    ) -> List[str]:
        picks = []
        SKIP_FLAG = ''
        DEBUG_FLAG = '\\debug'

        while True:
            guess = None
            while guess not in words and guess != SKIP_FLAG:
                guess = input("Your guess: ").strip().lower()
                if guess == DEBUG_FLAG:
                    print(my_words)
            

            if guess == SKIP_FLAG:
                break
            elif guess in my_words:
                picks.append(guess)
                print("Correct!")
            else:
                picks.append(guess)
                print("Wrong :(")
                break

        return picks

    def read_clue(self, word_set) -> Tuple[str, int]:
        while True:
            inp = input("Clue (e.g. 'car 2'): ").lower()
            match = re.match("(\w+)\s+(\d+)", inp)
            if match:
                clue, cnt = match.groups()
                if clue not in word_set:
                    print("I don't understand that word.")
                    continue
                return clue, int(cnt)

    def print_words(self, words: List[str], nrows: int):
        longest = max(map(len, words))
        print()
        for row in zip(*(iter(words),) * nrows):
            for word in row:
                print(word.rjust(longest), end=" ")
            print()
        print()


class Codenames:
    def __init__(self, cnt_rows=5, cnt_cols=5, cnt_agents=8, agg=.6):
        """
        :param cnt_rows: Number of rows to show.
        :param cnt_cols: Number of columns to show.
        :param cnt_agents: Number of good words.
        :param agg: Agressiveness in [0, infinity). Higher means more aggressive.
        """
        self.cnt_rows = cnt_rows
        self.cnt_cols = cnt_cols
        self.cnt_agents = cnt_agents
        self.agg = agg

        # Other
        self.vectors = np.array([])
        self.word_list = []
        self.weirdness = []
        self.word_to_index = {}
        self.codenames = []

    def load(self, datadir, limit_vocab=None):
        # stop words
        stopwords = set(w.lower().strip() for w in open(f"{datadir}/stopwords.txt"))
        # Glove word vectors
        print("...Loading words")
        self.word_list = [w.lower().strip() for w in open(f"{datadir}/words")]
        print("...Removing stopwords")
        stopwords_indices = [i for i, w in enumerate(self.word_list) if w in stopwords]
        self.word_list = np.array([w for w in self.word_list if w not in stopwords])

        print("...Loading vectors")
        self.vectors = np.load(f"{datadir}/glove.6B.300d.npy")

        if len(stopwords_indices) > 0:
            self.vectors = np.delete(self.vectors, stopwords_indices, axis=0)

        if limit_vocab is not None:
            self.word_list = self.word_list[:limit_vocab]
            self.vectors = self.vectors[:limit_vocab, :]

        assert len(self.word_list) == self.vectors.shape[0]
        # List of all glove words
        self.weirdness = [math.log(i + 1) + 1 for i in range(len(self.word_list))]

        # Indexing back from word to indices
        print("...Making word to index dict")
        self.word_to_index = {w: i for i, w in enumerate(self.word_list)}

        # All words that are allowed to go onto the table
        print("...Loading codenames")
        self.codenames: List[str] = [
            word
            for word in (w.lower().strip().replace(" ", "-") for w in open(f"{datadir}/wordlist"))
            if word in self.word_to_index
        ]

        print("Ready!")

    def word_to_vector(self, words: List[str]) -> np.ndarray:
        """
        :param words: To be vectorized.
        :return: The vector.
        """
        if isinstance(words, str):
            words = [words]
        else:
            assert isinstance(words, List)

        indices = [self.word_to_index[word] for word in words]
        return self.vectors[indices]

    def find_clue(
        self, words: List[str], my_words: List[str], used_clues: Iterable[str]
    ) -> Tuple[str, float, List[str]]:
        """
        :param words: Words on the board.
        :param my_words: Words we want to guess.
        :param used_clues: Clues we are not allowed to give.
        :return: (The best clue, the score, the words we expect to be guessed)
        """

        best_clue, best_score, best_group = '', 0., []

        most_count = 1
        max_value = 0.
        

        for candidate in self.candidates:
            if candidate not in used_clues:
                candidate_vector = self.word_to_vector(candidate)
                candidate_pos_values = candidate_vector @ self.word_to_vector(my_words).T
                current_count = np.sum(candidate_pos_values > SIMILARTY_THRESHOLD)
                
                if current_count >= most_count:
                    average_top_scores = np.mean(candidate_pos_values[candidate_pos_values > SIMILARTY_THRESHOLD])
                    if current_count > most_count:
                        best_score = 0.0
                    if average_top_scores > best_score:
                        best_score = average_top_scores
                        best_clue = candidate
                        most_count = current_count
                        best_group = [w for i, w in enumerate(my_words) if candidate_pos_values[0, i] > SIMILARTY_THRESHOLD]

        if most_count == 0:
            sys.exit('Ran out of candidates. Lower the threshold.')

        return best_clue, best_group

    def find_candidates(self, words, my_words):
        """
        Finds candidates by filtering the dissimilar words
        """
        pos_words = list(my_words)
        neg_words = list(set(words) - my_words)

        print("Thinking...")
        print("\n**While playing, hit enter to skip**")
        pos_values = self.vectors @ self.word_to_vector(pos_words).T
        pos_values_max = np.max(pos_values, axis = 1)

        neg_values = self.vectors @ self.word_to_vector(neg_words).T
        neg_values_max = np.max(neg_values, axis = 1)

        candidates = self.word_list[
            (pos_values_max > SIMILARTY_THRESHOLD) & 
            (neg_values_max < SIMILARTY_THRESHOLD)
            ]

        # removing words that are too similar to the positive words
        candidates = [
            c for c in candidates 
            if 
                (min([edit_distance(w, c) for w in pos_words]) > 2) and 
                (sum([similar_words(w, c) for w in pos_words]) == 0)
            ]

        return candidates

    def play_spymaster(self, reader: Reader):
        """
        Play a complete game, with the robot being the spymaster.
        """
        words = random.sample(self.codenames, self.cnt_rows * self.cnt_cols)
        my_words = set(random.sample(words, self.cnt_agents))
    
        self.candidates = self.find_candidates(words, my_words)

        used_clues = set(words)
        while my_words:
            reader.print_words(words, nrows=self.cnt_rows)

            clue, group = self.find_clue(words, list(my_words), used_clues)
            # Print the clue to the log_file for "debugging" purposes
            group_scores = self.word_to_vector(group) @ self.word_to_vector(clue).T
            print(clue, group, group_scores, file=log_file, flush=True)
            # Save the clue, so we don't use it again
            used_clues.add(clue)

            print()
            print(
                'Clue: "{} {}" (remaining words {})'.format(
                    clue, len(group), len(my_words)
                )
            )
            print()
            for pick in reader.read_picks(words, my_words, len(group)):
                words[words.index(pick)] = "---"
                if pick in my_words:
                    my_words.remove(pick)


def main():
    cn = Codenames()
    cn.load("dataset", limit_vocab=None)
    reader = TerminalReader()
    while True:
        try:
            mode = input("\nWill you be agent or spymaster?: ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        try:
            if mode == "spymaster":
                print("Not implemented yet.")
                # cn.play_agent(reader)
            elif mode == "agent":
                cn.play_spymaster(reader)
        except KeyboardInterrupt:
            # Catch interrupts from play functions
            pass


main()
