"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

-------------------------------------------------------------------

The multiset is a binary search tree implemented as a nested tuple tree,
together with the following methods

    insert: (multiset, symbol) -> multiset
    remove: (multiset, symbol) -> multiset

    forward_lookup: (multiset, symbol) -> (start, freq)
    reverse_lookup: (multiset, idx) -> (start, freq), symbol

where start/freq are the cumulative/frequency counts required to perform
ANS encode and decode.

See the README for an example.
"""
from functools import reduce
from math import floor

from craystack.codecs import Codec, substack
from craystack.rans import \
        push_with_finer_prec as rans_push, \
        pop_with_finer_prec as rans_pop, \
            base_message
from typing import Optional, Tuple, Union, Any, Sequence
from dataclasses import dataclass


def check_if_in_multiset(x, multiset):
    try:
        start, freq = forward_lookup(multiset, x)
        return freq > 0
    except ValueError:
        return False

def insert(multiset, x):
    '''Inserts a symbol x into the multiset'''
    size, y, m, left, right = multiset or (0, x, 0, (), ())
    if x < y:
        left = insert(left, x)
    elif x > y:
        right = insert(right, x)
    m = (left[2] if left else 0) + (right[2] if right else 0) + 1
    return size + 1, y, m, left, right

def remove(multiset, x):
    '''Removes a symbol x from the multiset'''
    size, y, m, left, right = multiset or (1, x, 1, (), ())
    if size == 1:
        return ()
    if x < y:
        left = remove(left, x)
    elif x > y:
        right = remove(right, x)
    m = (left[2] if left else 0) + (right[2] if right else 0) + 1
    return size - 1, y, m, left, right


def forward_lookup(multiset, x, bias=0):
    '''
    Looks up the cumulative (start) and frequency (freq) counts of symbol x.
    '''
    if not multiset:
        raise ValueError("The symbol {} could not be found.".format(x))
    size, y, m, left, right = multiset
    if x > y:
        start_right, freq = forward_lookup(right, x, bias)
        start = size - right[0] + start_right + (m - right[2])*bias
    elif x < y:
        start, freq = forward_lookup(left, x, bias)
    else:
        start = left[0] if left else 0
        freq = size - start - (right[0] if right else 0) + bias
    return start, freq


def reverse_lookup(multiset, idx, bias=0):
    '''
    Looks up the cumulative (start) and frequency (freq) counts,
    as well as the symbol x, at index idx.
    '''
    size, y, m, left, right = multiset or (0, (), 0, (), ())
    assert 0 <= idx < (size + m*bias), f'idx={idx} not in [0, {size + m*bias})'
    y_start = (left[0] + left[2]*bias) if left else 0
    y_freq = size - (left[0] if left else 0) - (right[0] if right else 0) + bias
    # print(f'y={y}, y_start={y_start}, y_freq={y_freq}, bias={bias}, idx={idx}')
    if idx < y_start:
        (start, freq), x = reverse_lookup(left, idx, bias)
    elif idx >= y_start + y_freq:
        size_not_right = (size + m*bias) - right[0] - right[2]*bias
        (start, freq), x = reverse_lookup(right, idx - size_not_right, bias)
        start = start + size_not_right
    else:
        x, start, freq = y, y_start, y_freq
    return (start, freq), x


def insert_then_forward_lookup(multiset, x):
    #TODO doesn't update the unique symbol count
    ''' Performs insert followed by forward_lookup, in one-pass.'''
    size, y, m, left, right = multiset or (0, x, 0, (), ())
    size = size + 1
    if x > y:
        right, (start_right, freq) = insert_then_forward_lookup(right, x)
        start = size - right[0] + start_right
    elif x < y:
        left, (start, freq) = insert_then_forward_lookup(left, x)
    else:
        start = left[0] if left else 0
        freq = size - start - (right[0] if right else 0)
    return (size, y, m, left, right), (start, freq)


def reverse_lookup_then_remove(multiset, idx):
    ''' Performs reverse_lookup followed by remove, in one-pass.'''
    size, y, m, left, right = multiset
    y_start = left[0] if left else 0
    y_freq = size - y_start - (right[0] if right else 0)
    if idx < y_start:
        left, (start, freq), x = reverse_lookup_then_remove(left, idx)
    elif idx >= y_start + y_freq:
        size_not_right = size - right[0]
        right, (start, freq), x = \
                reverse_lookup_then_remove(right, idx - size_not_right)
        start = start + size_not_right
    else:
        x, start, freq = y, y_start, y_freq
    size = size - 1
    return (size, y, m, left, right) if size else (), (start, freq), x


def build_multiset(sequence):
    '''Builds a multiset from the sequence by applying insert sequentially'''
    return tuple(reduce(insert, sequence, ()))


def to_sequence(multiset):
    ''' Flatten a BST, representing a multiset, to a sequence (python list)'''
    flat = []

    def traverse(branch):
        if branch:
            size, y, m, left, right = branch
            traverse(left)
            freq = size - (left[0] if left else 0) - (right[0] if right else 0)
            flat.extend(freq * [y])
            traverse(right)
    traverse(multiset)
    return flat


def check_multiset_equality(multiset, other_multiset):
    return sorted(to_sequence(multiset)) \
            == sorted(to_sequence(other_multiset))


def MultisetCodec(symbol_codec, multiset_size):
    '''
    Encodes a multiset using bits-back coding.
    Symbols are sampled from the multiset with SamplingWithoutReplacement,
    and pushd sequentially with symbol_codec.
    '''
    swor_codec = SamplingWithoutReplacementCodec()

    def push(ans_state, multiset):
        for _ in range(multiset_size):
            # 1) Sample, without replacement, a symbol using ANS pop.
            ans_state, symbol, multiset = \
                    swor_codec.pop(ans_state, multiset)

            # 2) Encode the selected symbol onto the same ANS state.
            ans_state = symbol_codec.push(ans_state, symbol)
        return ans_state

    def pop(ans_state):
        multiset = ()
        for _ in range(multiset_size):
            # Decode symbol on top of stack (reverses step 2)
            ans_state, symbol = symbol_codec.pop(ans_state)

            # Encode bits used to sample symbol (reverses step 1)
            # This is the bits-back step!
            ans_state, multiset = \
                    swor_codec.push(ans_state, symbol, multiset)
        return ans_state, multiset

    return Codec(push, pop)


def SamplingWithoutReplacementCodec():
    '''
    Encodes and pops onto the ANS state using the empirical
    distribution of symbols in the multiset.
    Before an push, the symbol to be pushd is inserted into the multiset.
    After a pop, the popd symbol is removed from the multiset. Therefore,
    a pop performs sampling without replacement, while push inverts it.
    The context is the multiset, i.e. *context = multiset
    '''
    def push(ans_state, symbol, multiset):
        multiset, (start, freq) = insert_then_forward_lookup(multiset, symbol)
        multiset_size = multiset[0]
        ans_state = rans_push(ans_state, start, freq, multiset_size)
        return ans_state, multiset

    def pop(ans_state, multiset):
        multiset_size = multiset[0]
        cdf_value, pop_ = rans_pop(ans_state, multiset_size)
        multiset, (start, freq), symbol = \
                reverse_lookup_then_remove(multiset, cdf_value[0])
        ans_state = pop_(start, freq)
        return ans_state, symbol, multiset

    return Codec(push, pop)


def FrequencyCountCodec(bias=0, precision=100):
    def push(ans_state, symbol, multiset):
        start, freq = forward_lookup(multiset, symbol, bias)
        multiset_size, _, num_unique_elements, *_ = multiset
        ans_state = rans_push(ans_state, start, freq, multiset_size + num_unique_elements*bias)
        return ans_state

    def pop(ans_state, multiset):
        multiset_size, _, num_unique_elements, *_ = multiset
        cdf_value, pop_ = rans_pop(ans_state, multiset_size + num_unique_elements*bias)
        (start, freq), symbol = reverse_lookup(multiset, cdf_value[0], bias)
        ans_state = pop_(start, freq)
        return ans_state, symbol

    return Codec(push, pop)


def compute_multiset_depth(multiset, depth=0):
    if multiset == ():
        return depth

    count, x, m, left, right = multiset
    left_depth = compute_multiset_depth(left, depth+1)
    right_depth = compute_multiset_depth(right, depth+1)
    return max(left_depth, right_depth)
