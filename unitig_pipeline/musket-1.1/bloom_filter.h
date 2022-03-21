/*
 *********************************************************************
 *                                                                   *
 *                           Open Bloom Filter                       *
 *                                                                   *
 * Author: Arash Partow - 2000                                       *
 * URL: http://www.partow.net                                        *
 * URL: http://www.partow.net/programming/hashfunctions/index.html   *
 *                                                                   *
 * Copyright notice:                                                 *
 * Free use of the Open Bloom Filter Library is permitted under the  *
 * guidelines and in accordance with the most current version of the *
 * Common Public License.                                            *
 * http://www.opensource.org/licenses/cpl1.0.php                     *
 *                                                                   *
 *********************************************************************
 */

#ifndef INCLUDE_BLOOM_FILTER_HPP
#define INCLUDE_BLOOM_FILTER_HPP

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <stdint.h>

static const std::size_t bits_per_char = 0x08; // 8 bits in 1 char(unsigned)
static const unsigned char bit_mask[bits_per_char] = { 0x01, //00000001
		0x02, //00000010
		0x04, //00000100
		0x08, //00001000
		0x10, //00010000
		0x20, //00100000
		0x40, //01000000
		0x80 //10000000
		};

class bloom_filter {
protected:

	typedef uint64_t bloom_type;
	typedef unsigned char cell_type;

public:

	bloom_filter(const std::size_t& predicted_element_count,
			const double& false_positive_probability,
			const std::size_t& random_seed) :
			bit_table_(0), predicted_element_count_(predicted_element_count), inserted_element_count_(
					0), random_seed_((random_seed) ? random_seed : 0xA5A5A5A5), desired_false_positive_probability_(
					false_positive_probability) {
		find_optimal_parameters();
		bit_table_ = new cell_type[table_size_ / bits_per_char];
		generate_unique_salt();
		std::fill_n(bit_table_, (table_size_ / bits_per_char), 0x00);
	}

	bloom_filter(const std::size_t& predicted_element_count,
			const std::size_t& bits_per_element, const std::size_t& random_seed) :
			bit_table_(0), predicted_element_count_(predicted_element_count), inserted_element_count_(
					0), random_seed_((random_seed) ? random_seed : 0xDEADBEEF) {
		table_size_ = bits_per_element * predicted_element_count_;
		bit_table_ = new cell_type[table_size_ / bits_per_char];
		salt_count_ = (std::size_t) (std::log(2.0) * bits_per_element);	/*this is a bug when bits_per_element = 1*/
		if(salt_count_ < 3) salt_count_ = 3;

		generate_unique_salt();
		std::fill_n(bit_table_, (table_size_ / bits_per_char), 0x00);
		desired_false_positive_probability_ = std::pow(
				1.0
						- std::exp(
								-1.0 * salt_.size() * predicted_element_count_
										/ size()), 1.0 * salt_.size());
	}

	bloom_filter(const bloom_filter& filter) {
		this->operator=(filter);
	}

	bloom_filter&
	operator =(const bloom_filter& filter) {
		salt_count_ = filter.salt_count_;
		table_size_ = filter.table_size_;
		predicted_element_count_ = filter.predicted_element_count_;
		inserted_element_count_ = filter.inserted_element_count_;
		random_seed_ = filter.random_seed_;
		desired_false_positive_probability_ =
				filter.desired_false_positive_probability_;
		delete[] bit_table_;
		bit_table_ = new cell_type[table_size_ / bits_per_char];
		std::copy(filter.bit_table_,
				filter.bit_table_ + (table_size_ / bits_per_char), bit_table_);
		salt_ = filter.salt_;
		return *this;
	}

	virtual ~bloom_filter() {
		delete[] bit_table_;
	}

	inline bool operator!() const {
		return (0 == table_size_);
	}

	inline void clear() {
		std::fill_n(bit_table_, (table_size_ / bits_per_char), 0x00);
		inserted_element_count_ = 0;
	}

	inline void insert(const unsigned char* key_begin,
			const std::size_t length) {
		std::size_t bit_index = 0;
		std::size_t bit = 0;
		for (std::vector<bloom_type>::iterator it = salt_.begin();
				it != salt_.end(); ++it) {
			compute_indices(hash_ap(key_begin, length, (*it)), bit_index, bit);
			bit_table_[bit_index / bits_per_char] |= bit_mask[bit];
		}
		++inserted_element_count_;
	}

	template<typename T>
	inline void insert(const T& t) {
		insert(reinterpret_cast<const unsigned char*>(&t), sizeof(T));
	}

	inline void insert(const std::string& key) {
		insert(reinterpret_cast<const unsigned char*>(key.c_str()), key.size());
	}

	inline void insert(const char* data, const std::size_t& length) {
		insert(reinterpret_cast<const unsigned char*>(data), length);
	}

	template<typename InputIterator>
	inline void insert(const InputIterator begin, const InputIterator end) {
		InputIterator it = begin;
		while (it != end) {
			insert(*(it++));
		}
	}

	inline virtual bool contains(const unsigned char* key_begin,
			const std::size_t length) const {
		std::size_t bit_index = 0;
		std::size_t bit = 0;
		for (std::vector<bloom_type>::const_iterator it = salt_.begin();
				it != salt_.end(); ++it) {
			compute_indices(hash_ap(key_begin, length, (*it)), bit_index, bit);
			if ((bit_table_[bit_index / bits_per_char] & bit_mask[bit])
					!= bit_mask[bit]) {
				return false;
			}
		}
		return true;
	}

	template<typename T>
	inline bool contains(const T& t) const {
		return contains(reinterpret_cast<const unsigned char*>(&t),
				static_cast<std::size_t>(sizeof(T)));
	}

	inline bool contains(const std::string& key) const {
		return contains(reinterpret_cast<const unsigned char*>(key.c_str()),
				key.size());
	}

	inline bool contains(const char* data, const std::size_t& length) const {
		return contains(reinterpret_cast<const unsigned char*>(data), length);
	}

	template<typename InputIterator>
	inline InputIterator contains_all(const InputIterator begin,
			const InputIterator end) const {
		InputIterator it = begin;
		while (it != end) {
			if (!contains(*it)) {
				return it;
			}
			++it;
		}
		return end;
	}

	template<typename InputIterator>
	inline InputIterator contains_none(const InputIterator begin,
			const InputIterator end) const {
		InputIterator it = begin;
		while (it != end) {
			if (contains(*it)) {
				return it;
			}
			++it;
		}
		return end;
	}

	inline virtual std::size_t size() const {
		return table_size_;
	}

	inline std::size_t element_count() const {
		return inserted_element_count_;
	}

	inline double desired_fpp() const {
		return desired_false_positive_probability_;
	}

	inline double effective_fpp() const {
		/*
		 Note:
		 The effective false positive probability is calculated using the
		 designated table size and hash function count in conjunction with
		 the current number of inserted elements - not the user defined
		 predicated/expected number of inserted elements.
		 */
		return std::pow(
				1.0
						- std::exp(
								-1.0 * salt_.size() * inserted_element_count_
										/ size()), 1.0 * salt_.size());
	}

	bloom_filter&
	operator &=(const bloom_filter& filter) {
		/* intersection */
		if ((salt_count_ == filter.salt_count_)
				&& (table_size_ == filter.table_size_)
				&& (random_seed_ == filter.random_seed_)) {
			for (std::size_t i = 0; i < (table_size_ / bits_per_char); ++i) {
				bit_table_[i] &= filter.bit_table_[i];
			}
		}
		return *this;
	}

	bloom_filter&
	operator |=(const bloom_filter& filter) {
		/* union */
		if ((salt_count_ == filter.salt_count_)
				&& (table_size_ == filter.table_size_)
				&& (random_seed_ == filter.random_seed_)) {
			for (std::size_t i = 0; i < (table_size_ / bits_per_char); ++i) {
				bit_table_[i] |= filter.bit_table_[i];
			}
		}
		return *this;
	}

	bloom_filter&
	operator ^=(const bloom_filter& filter) {
		/* difference */
		if ((salt_count_ == filter.salt_count_)
				&& (table_size_ == filter.table_size_)
				&& (random_seed_ == filter.random_seed_)) {
			for (std::size_t i = 0; i < (table_size_ / bits_per_char); ++i) {
				bit_table_[i] ^= filter.bit_table_[i];
			}
		}
		return *this;
	}

	const cell_type*
	table() const {
		return bit_table_;
	}

protected:

	inline virtual void compute_indices(const bloom_type& hash,
			std::size_t& bit_index, std::size_t& bit) const {
		bit_index = hash % table_size_;
		bit = bit_index % bits_per_char;
	}

	void generate_unique_salt() {
		/*
		 Note:
		 A distinct hash function need not be implementation-wise
		 distinct. In the current implementation "seeding" a common
		 hash function with different values seems to be adequate.
		 */
		const unsigned int predef_salt_count = 64;
		static const bloom_type predef_salt[predef_salt_count] = { 0xAAAAAAAA,
				0x55555555, 0x33333333, 0xCCCCCCCC, 0x66666666, 0x99999999,
				0xB5B5B5B5, 0x4B4B4B4B, 0xAA55AA55, 0x55335533, 0x33CC33CC,
				0xCC66CC66, 0x66996699, 0x99B599B5, 0xB54BB54B, 0x4BAA4BAA,
				0xAA33AA33, 0x55CC55CC, 0x33663366, 0xCC99CC99, 0x66B566B5,
				0x994B994B, 0xB5AAB5AA, 0xAAAAAA33, 0x555555CC, 0x33333366,
				0xCCCCCC99, 0x666666B5, 0x9999994B, 0xB5B5B5AA, 0xFFFFFFFF,
				0xFFFF0000, 0xB823D5EB, 0xC1191CDF, 0xF623AEB3, 0xDB58499F,
				0xC8D42E70, 0xB173F616, 0xA91A5967, 0xDA427D63, 0xB1E8A2EA,
				0xF6C0D155, 0x4909FEA3, 0xA68CC6A7, 0xC395E782, 0xA26057EB,
				0x0CD5DA28, 0x467C5492, 0xF15E6982, 0x61C6FAD3, 0x9615E352,
				0x6E9E355A, 0x689B563E, 0x0C9831A8, 0x6753C18B, 0xA622689B,
				0x8CA63C47, 0x42CC2884, 0x8E89919B, 0x6EDBD7D3, 0x15B6796C,
				0x1D6FDFE4, 0x63FF9092, 0xE7401432 };

		if (salt_count_ <= predef_salt_count) {
			std::copy(predef_salt, predef_salt + salt_count_,
					std::back_inserter(salt_));
			for (unsigned int i = 0; i < salt_.size(); ++i) {
				/*
				 Note:
				 This is done to integrate the user defined random seed,
				 so as to allow for the generation of unique bloom filter
				 instances.
				 */
				salt_[i] = salt_[i] * salt_[(i + 3) % salt_.size()]
						+ random_seed_;
			}
		} else {
			std::copy(predef_salt, predef_salt + predef_salt_count,
					std::back_inserter(salt_));
			srand(static_cast<unsigned int>(random_seed_));
			while (salt_.size() < salt_count_) {
				bloom_type current_salt = static_cast<bloom_type>(rand())
						* static_cast<bloom_type>(rand());
				if (0 == current_salt)
					continue;
				bool duplicate_found = false;
				for (std::vector<bloom_type>::iterator it = salt_.begin();
						it != salt_.end(); ++it) {
					if (current_salt == (*it)) {
						duplicate_found = true;
						break;
					}
				}
				if (!duplicate_found) {
					salt_.push_back(current_salt);
				}
			}
		}
	}

	void find_optimal_parameters() {
		/*
		 Note:
		 The following will attempt to find the number of hash functions
		 and minimum amount of storage bits required to construct a bloom
		 filter consistent with the user defined false positive probability
		 and estimated element insertion count.
		 */

		double min_m = std::numeric_limits<double>::infinity();
		double min_k = 0.0;
		double curr_m = 0.0;
		for (double k = 0.0; k < 1000.0; ++k) {
			if ((curr_m = ((-k * predicted_element_count_)
					/ std::log(
							1.0
									- std::pow(
											desired_false_positive_probability_,
											1.0 / k)))) < min_m) {
				min_m = curr_m;
				min_k = k;
			}
		}

		salt_count_ = static_cast<std::size_t>(min_k);
		table_size_ = static_cast<std::size_t>(min_m);
		table_size_ += (
				((table_size_ % bits_per_char) != 0) ?
						(bits_per_char - (table_size_ % bits_per_char)) : 0);
	}

	bloom_type hash_ap(const unsigned char* begin, std::size_t remaining_length,
			bloom_type hash) const {
		const unsigned char* it = begin;
		while (remaining_length >= 2) {
			hash ^= (hash << 7) ^ (*it++) * (hash >> 3);
			hash ^= (~((hash << 11) + ((*it++) ^ (hash >> 5))));
			remaining_length -= 2;
		}
		if (remaining_length) {
			hash ^= (hash << 7) ^ (*it) * (hash >> 3);
		}
		return hash;
	}

	std::vector<bloom_type> salt_;
	unsigned char* bit_table_;
	std::size_t salt_count_;
	std::size_t table_size_;
	std::size_t predicted_element_count_;
	std::size_t inserted_element_count_;
	std::size_t random_seed_;
	double desired_false_positive_probability_;
};

class compressible_bloom_filter: public bloom_filter {
public:

	compressible_bloom_filter(const std::size_t& predicted_element_count,
			const double& false_positive_probability,
			const std::size_t& random_seed) :
			bloom_filter(predicted_element_count, false_positive_probability,
					random_seed) {
		size_list.push_back(table_size_);
	}

	inline virtual std::size_t size() const {
		return size_list.back();
	}

	inline bool compress(const double& percentage) {
		if ((0.0 >= percentage) || (percentage >= 100.0))
			return false;
		std::size_t original_table_size = size_list.back();
		std::size_t new_table_size = static_cast<std::size_t>((size_list.back()
				* (1.0 - (percentage / 100.0))));
		new_table_size -= (
				((new_table_size % bits_per_char) != 0) ?
						(new_table_size % bits_per_char) : 0);
		if ((bits_per_char > new_table_size)
				|| (new_table_size >= original_table_size))
			return false;
		desired_false_positive_probability_ = effective_fpp();
		cell_type* tmp = new cell_type[new_table_size / bits_per_char];
		std::copy(bit_table_, bit_table_ + (new_table_size / bits_per_char),
				tmp);
		cell_type* it = bit_table_ + (new_table_size / bits_per_char);
		cell_type* end = bit_table_ + (original_table_size / bits_per_char);
		cell_type* it_tmp = tmp;
		while (it != end) {
			*(it_tmp++) |= (*it++);
		}
		delete[] bit_table_;
		bit_table_ = tmp;
		size_list.push_back(new_table_size);
		return true;
	}

private:

	inline virtual void compute_indices(const bloom_type& hash,
			std::size_t& bit_index, std::size_t& bit) const {
		bit_index = hash;
		for (unsigned int j = 0; j < size_list.size(); bit_index %=
				size_list[j++])
			;
		bit = bit_index % bits_per_char;
	}

	std::vector<std::size_t> size_list;
};

#endif

/*
 Note 1:
 If it can be guaranteed that bits_per_char will be of the form 2^n then
 the following optimization can be used:

 hash_table[bit_index >> n] |= bit_mask[bit_index & (bits_per_char - 1)];

 Note 2:
 For performance reasons where possible when allocating memory it should
 be aligned (aligned_alloc) according to the architecture being used.
 */
