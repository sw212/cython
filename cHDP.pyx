# cython: language_level=3, binding=True, boundscheck=False, cdivision=True, initializedcheck=False, wraparound=False
# for detailed debug: linetrace
# distutils: language = c++

cimport cython
from cpython cimport array
import array
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport gammaln
from libc.math cimport exp, log
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.limits cimport LONG_MAX
from cython.operator cimport dereference as deref, preincrement as inc

cdef class Document:
    cdef readonly np.int_t [:] words
    cdef public np.int_t [::1] table_by_i
    cdef public np.int_t [::1] topic_by_table
    cdef public vector[unordered_map[np.int_t, np.int_t]] num_words_by_tw
    cdef public np.int_t [::1] num_words_by_t
    cdef public vector[np.int_t] using_tables
    cdef public vector[np.int_t] vacant_tables
    cdef public np.int_t num_tables

    def __init__(self, list doc, np.int_t size_vocab):
        self.words = np.array(doc, dtype=np.int_)
        self.table_by_i = np.full_like(self.words, 0)
        self.topic_by_table = np.array([0], dtype=np.int_)
        self.num_words_by_t = np.array([0], dtype=np.int_)
        self.num_words_by_tw = [{}]
        self.using_tables = []
        self.vacant_tables = [0]
        self.num_tables = 0


cdef class HDP:
    cdef public np.int_t size_vocab
    cdef public Document[:] docs
    cdef Document d

    cdef public vector[unordered_map[np.int_t, np.int_t]] num_words_by_kw
    cdef public np.int_t [::1] num_words_by_k
    cdef public np.int_t [::1] num_tables_by_k

    cdef public vector[np.int_t] using_topics
    cdef public vector[np.int_t] vacant_topics
    cdef public np.int_t global_num_tables
    cdef public np.int_t num_topics

    cdef np.float64_t [::1] p
    cdef np.float64_t [::1] q
    cdef np.float64_t [::1] f

    cdef public np.float64_t alpha
    cdef public np.float64_t gamma
    cdef public np.float64_t eta

    cdef object rng

    def __init__(self, list documents, float alpha=1.0, float gamma=1.0, float eta=0.5, seed=None):
        self.size_vocab = max(max(doc) for doc in documents) + 1
        self.docs = np.array([Document(doc, self.size_vocab) for doc in documents])

        self.num_words_by_kw = [{}]
        self.num_words_by_k = np.array([0], dtype=np.int_)
        self.num_tables_by_k = np.array([0], dtype=np.int_)

        self.using_topics = []
        self.vacant_topics = [0]
        self.global_num_tables = 0
        self.num_topics = 0

        self.p = np.zeros(2, dtype=np.float64)
        self.q = np.zeros(2, dtype=np.float64)
        self.f = np.zeros(2, dtype=np.float64)

        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta

        self.rng = np.random.default_rng(seed)

        self.init_state()

    cdef void init_state(self):
        cdef np.int_t j, i
        cdef np.int_t w, t_ind, k_ind
        cdef np.float64_t [::1] f, q
        cdef np.float64_t u
        for j in range(self.docs.shape[0]):
            self.d = self.docs[j]
            for i in range(self.d.words.shape[0]):
                if not j and not i:
                    self.assign_table(0, 0, 0, 0)
                    continue
                elif j and not i:
                    w = self.d.words[i]
                    self.compute_f(w)
                    self.word_topic_probabilities()
                    u = self.rng.random()
                    k_ind = self.get_sampled_idx(self.q, u, 1+self.num_topics)
                    self.assign_table(j, i, 0, k_ind)
                    continue
        
                t_ind, k_ind = self.sample_table(j, i)
                self.assign_table(j, i, t_ind, k_ind)

    cdef void remove_word(self, np.int_t j, np.int_t i):
        cdef np.int_t w, t, k
        w = self.d.words[i]
        t = self.d.table_by_i[i]
        k = self.d.topic_by_table[t]

        self.d.num_words_by_tw[t][w] -= 1
        self.d.num_words_by_t[t] -= 1
        self.num_words_by_k[k] -= 1
        self.num_words_by_kw[k][w] -= 1

        if not self.num_words_by_kw[k][w]:
            self.num_words_by_kw[k].erase(w)
        if not self.d.num_words_by_tw[t][w]:
            self.d.num_words_by_tw[t].erase(w)

        if not self.d.num_words_by_t[t]:
            self.d.num_tables -= 1
            self.global_num_tables -= 1
            self.num_tables_by_k[k] -= 1
            self.remove(&self.d.using_tables, t)
            self.d.vacant_tables.push_back(t)

            if not self.num_tables_by_k[k]:
                self.num_topics -= 1
                self.remove(&self.using_topics, k)
                self.vacant_topics.push_back(k)

    cdef void remove_topic(self, np.int_t j, np.int_t t):
        cdef np.int_t k, w, count
        k = self.d.topic_by_table[t]
        self.num_tables_by_k[k] -= 1
        self.num_words_by_k[k] -= self.d.num_words_by_t[t]

        if not self.num_tables_by_k[k]:
            assert not self.num_words_by_k[k]
            self.num_topics -= 1
            self.remove(&self.using_topics, k)
            self.vacant_topics.push_back(k)

        cdef unordered_map[np.int_t,np.int_t].iterator it = self.d.num_words_by_tw[t].begin()
        cdef unordered_map[np.int_t,np.int_t].iterator end = self.d.num_words_by_tw[t].end()
        while it != end:
            w = deref(it).first
            count = deref(it).second
            self.num_words_by_kw[k][w] -= count
            if not self.num_words_by_kw[k][w]:
                self.num_words_by_kw[k].erase(w)
            inc(it)

    cdef void compute_f(self, np.int_t w):
        cdef np.int_t i, k
        cdef np.float64_t eta_vocab = (self.eta * self.size_vocab)
        for i in range(self.num_topics):
            k = self.using_topics[i]
            self.f[i] = (self.eta + self.num_words_by_kw[k][w]) / (self.num_words_by_k[k] + eta_vocab)

    cdef void word_table_probabilities(self, np.int_t j):
        cdef np.float64_t f_new, p_tot
        cdef np.int_t i, t, k, k_ind

        f_new = self.gamma / self.size_vocab
        for i in range(self.num_topics):
            k = self.using_topics[i]
            f_new += self.num_tables_by_k[k] * self.f[i]
        f_new /= self.gamma + self.global_num_tables

        p_tot = 0
        for i in range(self.d.num_tables):
            t = self.d.using_tables[i]
            k = self.d.topic_by_table[t]
            k_ind = self.index(&self.using_topics, k)
            p_tot += self.d.num_words_by_t[t] * self.f[k_ind]
            self.p[i] = p_tot
        p_tot += f_new
        self.p[self.d.num_tables] = p_tot

        for i in range(1 + self.d.num_tables):
            self.p[i] /= p_tot

    cdef void word_topic_probabilities(self):
        cdef np.float64_t q_tot = 0
        cdef np.int_t i, k

        for i in range(self.num_topics):
            k = self.using_topics[i]
            q_tot += self.num_tables_by_k[k] * self.f[i]
            self.q[i] = q_tot
        q_tot += self.gamma / self.size_vocab
        self.q[self.num_topics] = q_tot

        for i in range(1 + self.num_topics):
            self.q[i] /= q_tot

    cdef (np.int_t, np.int_t) sample_table(self, np.int_t j, np.int_t i):
        cdef np.int_t w
        cdef np.float64_t u
        cdef np.int_t t, k, t_ind, k_ind

        w = self.d.words[i]
        self.compute_f(w)
        self.word_table_probabilities(j)

        u = self.rng.random()
        t_ind = self.get_sampled_idx(self.p, u, 1+self.d.num_tables)
        k_ind = 0

        if t_ind == self.d.num_tables:
            self.word_topic_probabilities()
            u = self.rng.random()
            k_ind = self.get_sampled_idx(self.q, u, 1+self.num_topics)
        else:
            t = self.d.using_tables[t_ind]
            k = self.d.topic_by_table[t]
            k_ind = self.index(&self.using_topics, k)
        return t_ind, k_ind

    cdef np.int_t get_sampled_idx(self, np.float64_t [::1] arr, np.float64_t u, np.int_t size):
        cdef np.int_t i
        for i in range(size):
            if u < arr[i]:
                return i

    cdef void table_topic_probabilities(self, np.int_t j, np.int_t t):
        cdef np.float64_t eta_V, q_max
        cdef np.int_t w, k_ind, k
        cdef unordered_map[np.int_t,np.int_t].iterator it = self.d.num_words_by_tw[t].begin()
        cdef unordered_map[np.int_t,np.int_t].iterator end = self.d.num_words_by_tw[t].end()

        eta_V = self.eta * self.size_vocab
        self.q[self.num_topics] = log(self.gamma) + gammaln(eta_V) - gammaln(eta_V +self.d.num_words_by_t[t])

        while it != end:
            self.q[self.num_topics] += gammaln(self.eta + deref(it).second) - gammaln(self.eta)
            inc(it)

        it = self.d.num_words_by_tw[t].begin()
        for k_ind in range(self.num_topics):
            k = self.using_topics[k_ind]
            self.q[k_ind] = log(self.num_tables_by_k[k])
            self.q[k_ind] += gammaln(self.num_words_by_k[k] + eta_V) - gammaln(self.num_words_by_k[k] + self.d.num_words_by_t[t] + eta_V)

            it = self.d.num_words_by_tw[t].begin()
            while it != end:
                w = deref(it).first
                self.q[k_ind] += gammaln(self.num_words_by_kw[k][w] + deref(it).second + self.eta) - gammaln(self.num_words_by_kw[k][w] + self.eta)
                inc(it)

        q_max = np.max(self.q[:1+self.num_topics])
        for k_ind in range(self.num_topics + 1):
            self.q[k_ind] = exp(self.q[k_ind] - q_max)
            if k_ind > 0:
                self.q[k_ind] += self.q[k_ind-1]

        q_max = self.q[self.num_topics]
        for k_ind in range(self.num_topics + 1):
            self.q[k_ind] /= q_max

    cdef np.int_t sample_topic(self, np.int_t j, np.int_t t):
        cdef np.float64_t u
        self.table_topic_probabilities(j, t)
        u = self.rng.random()
        return self.get_sampled_idx(self.q, u, 1+self.num_topics)

    cdef void assign_topic(self, np.int_t j, np.int_t t, np.int_t k_ind):
        cdef np.int_t k, w 
        if k_ind == self.num_topics:
            self.num_topics += 1

            if self.vacant_topics.empty():
                self.add_topics()

            k = self.pop_min(&self.vacant_topics)
            k_ind = self.insert_sorted(&self.using_topics, k)
        else:
            k = self.using_topics[k_ind]

        self.d.topic_by_table[t] = k
        self.num_tables_by_k[k] += 1
        self.num_words_by_k[k] += self.d.num_words_by_t[t]

        cdef unordered_map[np.int_t,np.int_t].iterator it = self.d.num_words_by_tw[t].begin()
        cdef unordered_map[np.int_t,np.int_t].iterator end = self.d.num_words_by_tw[t].end()
        while it != end:
            w = deref(it).first
            self.num_words_by_kw[k][w] += deref(it).second
            inc(it)

    cdef void assign_table(self, np.int_t j, np.int_t i, np.int_t t_ind, np.int_t k_ind):
        cdef np.int_t w, t, k

        w = self.d.words[i]

        if k_ind == self.num_topics:
            self.num_topics += 1

            if self.vacant_topics.empty():
                self.add_topics()

            k = self.pop_min(&self.vacant_topics)
            k_ind = self.insert_sorted(&self.using_topics, k)
        else:
            k = self.using_topics[k_ind]

        if t_ind == self.d.num_tables:
            self.num_tables_by_k[k] += 1
            self.d.num_tables = self.d.num_tables + 1
            self.global_num_tables += 1

            if self.d.vacant_tables.empty():
                self.add_tables(j)

            t = self.pop_min(&self.d.vacant_tables)
            t_ind = self.insert_sorted(&self.d.using_tables, t)

        t = self.d.using_tables[t_ind]
        k = self.using_topics[k_ind]

        self.d.topic_by_table[t] = k
        self.d.table_by_i[i] = t
        self.d.num_words_by_tw[t][w] += 1
        self.d.num_words_by_t[t] += 1
        self.num_words_by_kw[k][w] += 1
        self.num_words_by_k[k] += 1

    cdef np.int_t insert_sorted(self, vector[np.int_t]* arr, np.int_t num):
        if arr.size() == 0:
            arr.push_back(num)
            return 0
        cdef np.int_t i = 0
        cdef vector[np.int_t].iterator it = arr.begin()
        cdef vector[np.int_t].iterator end = arr.end()
        while it != end:
            if num < deref(it):
                arr.insert(it, num)
                return i
            inc(it)
            i += 1
        arr.push_back(num)
        return i
    
    cdef np.int_t index(self, vector[np.int_t]* arr, np.int_t num):
        cdef vector[np.int_t].iterator it = arr.begin()
        cdef vector[np.int_t].iterator end = arr.end()
        cdef np.int_t i = 0
        while it != end:
            if deref(it) == num:
                return i
            inc(it)
            i += 1
        raise ValueError('index: num not found')
    
    cdef void remove(self, vector[np.int_t]* arr, np.int_t num):
        cdef vector[np.int_t].iterator it = arr.begin()
        cdef vector[np.int_t].iterator end = arr.end()
        while it != end:
            if num == deref(it):
                arr.erase(it)
                return
            inc(it)
        raise ValueError(f'remove(num): num not in array')
    
    cdef np.int_t pop_min(self, vector[np.int_t]* arr):
        cdef vector[np.int_t].iterator it = arr.begin()
        cdef vector[np.int_t].iterator end = arr.end()
        cdef vector[np.int_t].iterator loc
        cdef np.int_t val = LONG_MAX
        while it != end:
            if deref(it) < val:
                val = deref(it)
                loc = it
            inc(it)
        if val > -1:
            arr.erase(loc)
            return val
        raise ValueError('pop_min: no minimum found')

    cdef void add_topics(self):
        cdef np.int_t num_topics = self.num_tables_by_k.shape[0]
        cdef np.int_t i
        self.f = np.hstack((self.f, np.zeros_like(self.f)))
        self.q = np.hstack((self.q, np.zeros_like(self.q)))

        for i in range(num_topics, 2*num_topics):
            self.vacant_topics.push_back(i)

        for i in range(num_topics):
            self.num_words_by_kw.push_back(unordered_map[np.int_t, np.int_t]())
        self.num_words_by_k = np.hstack((self.num_words_by_k, np.zeros_like(self.num_words_by_k)))
        self.num_tables_by_k = np.hstack((self.num_tables_by_k, np.zeros_like(self.num_tables_by_k)))

    cdef void add_tables(self, np.int_t j):
        cdef np.int_t max_table = 0
        cdef np.int_t n_tables = self.d.topic_by_table.shape[0]
        cdef np.int_t i
        for j in range(self.docs.shape[0]):
            if self.docs[j].topic_by_table.shape[0] > max_table:
                max_table = self.docs[j].topic_by_table.shape[0]
        if self.p.shape[0] < 1 + (2 * max_table):
            self.p = np.hstack((self.p, np.zeros_like(self.p)))

        for i in range(n_tables):
            self.d.num_words_by_tw.push_back(unordered_map[np.int_t, np.int_t]())

        for i in range(n_tables, 2*n_tables):
            self.d.vacant_tables.push_back(i)

        self.d.topic_by_table = np.hstack((self.d.topic_by_table, np.zeros_like(self.d.topic_by_table)))
        self.d.num_words_by_t = np.hstack((self.d.num_words_by_t, np.zeros_like(self.d.num_words_by_t)))

    def gibbs_step(self):
        cdef np.int_t j, i, t_ind, k_ind
        for j in range(self.docs.shape[0]):
            self.d = self.docs[j]
            for i in range(self.d.words.shape[0]):
                self.remove_word(j, i)
                t_ind, k_ind = self.sample_table(j, i)
                self.assign_table(j, i, t_ind, k_ind)

            for t_ind in range(self.d.num_tables):
                t = self.d.using_tables[t_ind]
                self.remove_topic(j, t)
                k_ind = self.sample_topic(j, t)
                self.assign_topic(j, t, k_ind)