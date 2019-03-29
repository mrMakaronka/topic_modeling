import io
import numpy as np


class FileReadUtil:
    @staticmethod
    def load_fasttext(path, skip=0, limit=int(10e9)) -> (np.ndarray, np.ndarray):
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n, d = [int(x) for x in fin.readline().split()]
            n = int(min(n, limit))

            data = np.zeros((n, d), dtype=np.float32)
            words = np.empty(n, dtype=object)

            i = 0
            for line in fin:
                if i >= skip:
                    tokens = line.split()
                    data[i - skip] = np.float32(tokens[1:])
                    words[i - skip] = tokens[0]

                i += 1
                if (i - skip) == limit:
                    break
            return data, words

    @staticmethod
    def load_cluster_centroids(path) -> np.ndarray:
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n, d = [int(x) for x in fin.readline().split()]
            data = np.zeros((n, d), dtype=np.float32)
            i = 0
            for line in fin:
                data[i] = np.float32(line.split())
                i += 1
            return data

    @staticmethod
    def load_clusters_names(path) -> np.ndarray:
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n = int(fin.readline())
            words = np.empty(n, dtype=object)
            i = 0
            for line in fin:
                words[i] = line
                i += 1
            return words

    @staticmethod
    def load_topics_matching(path) -> np.ndarray:
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n = int(fin.readline())
            topics = np.empty(n, dtype=object)
            i = 0
            for line in fin:
                split = line.replace('\r\n', '').split('\t')
                if len(split) == 2:
                    topics[i] = split[1].split(';')
                else:
                    topics[i] = list()
                i += 1
            return topics

    @staticmethod
    def load_idf(path) -> np.ndarray:
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            n = int(fin.readline())
            idf = np.empty(n, dtype=np.float32)
            i = 0
            for line in fin:
                idf[i] = float(line)
                i += 1
            return idf
