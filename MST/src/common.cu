#ifndef common
#define common
#define DEBUG false
#include "cuda_timer.cu"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <limits>

typedef uint32_t Vertex;
typedef uint64_t Weight;
const Weight MAX_WT = UINTMAX_MAX;
#define CSR_SIZE(n, m) (sizeof(Vertex) * (n + m + 1)) + (sizeof(Weight) * m)
// #define MALLOC(p, q) (p*) malloc(q * sizeof(p))


/**
 * @brief Ensures that multiplication
 * does not overflow/wrap, otherwise returns 0
 * @param x value to multiply
 * @param y value to multiply
 * @return Weight 0 if wrap ouccers, otherwise 
 * returns result of multiplication
 */
Weight SafeMultiplication(Weight x, Weight y)
{
	if(x > MAX_WT/ sizeof(y))
		return 0;
	return x * y;
}

/**
 * @brief Ensures that addition
 * does not overflow/wrap, otherwise returns 0
 * @param x value to multiply
 * @param y value to multiply
 * @return Weight 0 if wrap ouccers, otherwise 
 * returns result of addition
 */
Weight SafeAddition(Weight x, Weight y)
{
	if(MAX_WT - x < y)
		return 0;
	return x + y;
}

template <typename T>
T* MALLOC(size_t size)
{
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wall"
	size = SafeMultiplication(size, sizeof(T)); // ignore
	#pragma clang diagnostic pop
	if(size == 0) return nullptr;
	return (T*)malloc(size);
}

#pragma region GPU_Utility

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t code, const char* file, int line,
					  bool abort = true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", 
				cudaGetErrorString(code), file, line);
		if(abort)
			exit(code);
	}
}

#define BLOCKSIZE 1024

void PrintByteSize(double bytes)
{
	std::cout << bytes << " bytes";
	if(bytes >= 0x400)
	{
		bytes /= 0x400;
		std::cout << " =  " << bytes << " KB";
	}
	if(bytes >= 0x400)
	{
		bytes /= 0x400;
		std::cout << " =  " << bytes << " MB";
	}
	if(bytes >= 0x400)
	{
		bytes /= 0x400;
		std::cout << " =  " << bytes << " GB";
	}
}

#pragma endregion GPU_Utility

#pragma region random_wrapper_Region

/**
 * @brief Object to generate pseudorandom numbers
 * with the given setting
 */
class random_wrapper
{
	std::random_device					  rd;
	std::mt19937						  mt;
	std::uniform_int_distribution<Vertex> random_gen;

  public:
	/**
	 * @brief Construct a new random wrapper object
	 *
	 * @param low The lowest value
	 * that this object should generate randomly
	 * @param high The highest value that this
	 * object should generate randomly
	 */
	random_wrapper(Vertex low, Vertex high)
		: rd(), mt(rd()), random_gen(low, high)
	{
	}

	/**
	 * @brief Generates a pseudorandom number
	 *
	 * @return Vertex The pseudorandom value
	 */
	Vertex generate() { return random_gen(mt); }

	/**
	 * @brief Generates and fills an entire array
	 * with random values
	 *
	 * @param array The array to fill
	 * @param size The size of array to fill
	 */
	void generateFill(Vertex* array, Vertex size)
	{
		for(Vertex* a = array; size > 0; size--, a++)
		{
			*a = generate();
		}
	}
};

#pragma endregion random_wrapper_Region

#pragma region Array_Region

namespace CustomArray
{
	/**
	 * @brief My custom array implementation
	 *
	 * @tparam T Generic parameter
	 */
	template <typename T> class Array1D
	{
		/**
		 * @brief Inner array
		 *
		 */
		T* array;
		/**
		 * @brief The size of the array
		 *
		 */
		Vertex count;

	  public:
		/**
		 * @brief Construct a new Array 1 D object
		 *
		 * @param size : The size of the array
		 * @param call_const : if true, constructor is called on each element
		 */
		Array1D(int size)
		{
			if(size < 1)
			{
				size  = 0;
				array = nullptr;
			}
			else
			{
				count = size;
				array = MALLOC<T>(count);
			}
		}

		/**
		 * @brief Construct a new Array 1D object
		 * by making a deep copy of the array provided
		 *
		 * @param data The array to make a copy of
		 * @param size The size of the array
		 * @param copy Should a copy be made or not.
		 * If not, expects an array allocated using malloc
		 */
		Array1D(T* data, int size, bool copy=true)
		{
			if(size < 1)
			{
				count = 0;
				array = nullptr;
			}
			else if(copy)
			{
				count = size;
				array = MALLOC<T>(count);
				for(int i = 0; i < size; i++)
					array[i] = data[i];
			}
			else
			{
				count = size;
				array = data;
			}
		}

		/**
		 * @brief Construct a new Array 1D object
		 * from a given initializer list
		 *
		 * @param list initializer list to populate
		 * the array
		 */
		Array1D(std::initializer_list<T> list)
		{
			count = list.size();
			array = MALLOC<T>(count);
			int p = 0;
			for(T x: list)
			{
				array[p++] = x;
			}
		}

		/**
		 * @brief Copy Constructor for Array1D
		 *
		 * @param other the Array1D object to copy
		 */
		Array1D(const Array1D<T>& other)
		{
			count = other.count;
			if(count <= 0)
			{
				count = 0;
				array = nullptr;
				return;
			}
			array = MALLOC<T>(count);
			for(Vertex i = 0; i < count; i++)
			{
				array[i] = other.array[i];
			}
		}

		/**
		 * @brief Move Constructor for Array 1D
		 *
		 * @param other the Array1D object to move
		 */
		Array1D(Array1D<T>&& other)
		{
			count		= other.count;
			array		= other.array;
			other.array = nullptr;
			other.count = 0;
		}

		/**
		 * @brief Get the size of the array
		 *
		 * @return int : array size
		 */
		Weight GetCount() const { return count; }

		/**
		 * @brief Sets an item at a given position
		 *
		 * @param pos position
		 * @param item element to insert in the array
		 */
		void Set(int pos, T item) { array[pos] = item; }

		/**
		 * @brief Gets an element from the array at the given position
		 *
		 * @param pos The position
		 * @return T The item from the array
		 */
		T Get(int pos) const { return array[pos]; }

		/**
		 * @brief Iterates every element in the array and executes the given
		 * function
		 *
		 * @param itr The function to execute for each element
		 */
		void Iterate(std::function<void(const T)> itr)
		{
			T *a = array, *e = array + count;
			while(a < e)
			{
				itr(*a);
				a++;
			}
		}

		/**
		 * @brief Iterates every element in the array and executes the given
		 * function
		 *
		 * @param itr The function to execute for each element
		 */
		void Iterate(std::function<void(const T, const int pos)> itr)
		{
			for(int i = 0, *a = array; i < count; i++, a++)
				itr(*a, i);
		}

		/**
		 * @brief Calls the std::sort in the internal
		 * array using the default comparator
		 *
		 */
		void Sort() { std::sort(array, array + count); }

		T* Ref() { return array; }

		/**
		 * @brief Iterates and finds the index of the first
		 * element that satisfies the given constraint, if
		 * it exists, otherwise returns -1
		 *
		 * @param cond The condition to satisfy
		 * @param start The index of the array
		 * to start searching from
		 * @return int index of the element if found,
		 * else -1
		 */
		int64_t FindFirst(std::function<bool(const T)> cond, int start = 0)
		{
			int64_t x = count;
			for(int64_t i = start; i < x; i++)
			{
				if(cond(array[i]))
					return i;
			}
			return -1;
		}

		/**
		 * @brief Copy assignment operator
		 *
		 * @param other
		 * @return Array1D&
		 */
		Array1D& operator=(const Array1D<T>& other)
		{
			return *this = Array1D(other);
		}

		/**
		 * @brief Move assignment via operator
		 *
		 * @param other The array to assign
		 * @return Array1D& The returned reference
		 */
		Array1D& operator=(Array1D<T>&& other)
		{
			array		= other.array;
			count		= other.array;
			other.array = nullptr;
			other.count = 0;
			return *this;
		}

		/**
		 * @brief Compares two arrays and
		 * checks if they are equivalent or not
		 *
		 * @param other The array to compare
		 * @return bool if true, then both arrays
		 * are equivalent, otherwise false
		 */
		bool operator==(const Array1D<T>& other) const
		{
			if(count != other.count)
				return false;
			for(int i = 0; i < count; i++)
			{
				if(Get(i) != other.Get(i))
					return false;
			}
			return true;
		}

		/**
		 * @brief Compares two arrays and
		 * checks if they are equivalent or not
		 *
		 * @param other The array to compare
		 * @return bool if false, then both arrays
		 * are equivalent, otherwise true
		 */
		bool operator!=(const Array1D<T>& other) const
		{
			return !(*this == other);
		}

		/**
		 * @brief Destroy the Array1D object
		 *
		 */
		~Array1D()
		{
			if(array != nullptr)
				free(array);
		}
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, Array1D<T> array)
	{
		int count;
		if((count = array.GetCount()) < 1)
		{
			os << "[ ]";
			return os;
		}
		os << "[ " << array.Get(0);
		for(int i = 1; i < count; i++)
		{
			os << ", " << array.Get(i);
		}
		os << " ]";
		return os;
	}

	/**
	 * @brief My custom implementation of a 2D array
	 *
	 * @tparam T Generic parameter
	 */
	template <typename T> class Array2D
	{
		/**
		 * @brief The inner array maintained
		 * by the class
		 *
		 */
		T* array;
		/**
		 * @brief Dimensions of the 2D array
		 *
		 */
		int row, col;

	  public:
		/**
		 * @brief Construct a new Array 2 D object
		 *
		 * @param r Rows in this 2D array
		 * @param c Columns in this 2D array
		 * @param call_const if true, the
		 * constructor is called on each element
		 */
		Array2D(int r, int c)
		{
			if(r <= 0 || c <= 0)
			{
				r = c = 0;
				array = nullptr;
			}
			else
			{
				row = r;
				col = c;
				array = MALLOC<T>(r * c);
			}
		}

		/**
		 * @brief Construct a new Array 1D object
		 * by making a deep copy of the array provided
		 *
		 * @param a The array to make a copy of
		 * @param r Number of rows in the array
		 * @param c Number of columns in the array
		 */
		Array2D(int r, int c, T** array)
		{
			if(r <= 0 || c <= 0)
			{
				r = c = 0;
				array = nullptr;
			}
			else
			{
				row = r;
				col = c;
				int i, j;
				array = MALLOC<T>(r*c);
				for(i = 0; i < r; i++)
				{
					for(j = 0; j < c; j++)
					{
						Set(i, j, array[i][j]);
					}
				}
			}
		}

		/**
		 * @brief Copy Constructor for Array2D
		 *
		 * @param other The Array2D object to copy from
		 */
		Array2D(const Array2D<T>& other)
		{
			row = other.row;
			col = other.col;
			if(row <= 0 || col <= 0)
			{
				array = nullptr;
				return;
			}
			array = MALLOC<T>(row * col);
			int i, j;
			for(i = 0; i < row; i++)
			{
				for(j = 0; j < col; j++)
				{
					Set(i, j, other.array[(i * col) + j]);
				}
			}
		}

		/**
		 * @brief Move Constructor for Array2D
		 *
		 * @param other The object to move data from
		 */
		Array2D(Array2D<T>&& other)
		{
			row			= other.row;
			col			= other.col;
			array		= other.array;
			other.array = nullptr;
		}

		/**
		 * @brief Construct a new Array2D object
		 * from a nested initializer list
		 *
		 * @param list The nested initializer list
		 */
		Array2D(std::initializer_list<std::initializer_list<T>> list)
		{
			row = list.size();
			if(row >= 1 && (col = list.begin()->size()) >= 1)
			{
				array	  = MALLOC<T>(row * col);
				int pos_i = 0;
				int pos_j = 0;
				for(auto i = list.begin(); i != list.end(); ++i)
				{
					for(auto j = i->begin(); j != i->end(); ++j)
					{
						Set(pos_i, pos_j, *j);
						pos_j++;
					}
					pos_i++;
					pos_j = 0;
				}
			}
			else
			{
				row = col = 0;
			}
		}

		/**
		 * @brief Get the number of Rows
		 *
		 * @return int: The number of rows
		 */
		int GetRowCount() { return row; }

		/**
		 * @brief Get the number of Columns
		 *
		 * @return int: The number of columns
		 */
		int GetColumnCount() { return col; }

		/**
		 * @brief Sets an element at the given position
		 *
		 * @param r The row of the element to set
		 * @param c The column of the element to set
		 * @param item The item to set at the given position
		 */
		void Set(int r, int c, T item) { array[(r * col) + c] = item; }

		/**
		 * @brief Gets an element from the given position
		 *
		 * @param r the row of this element
		 * @param c the column of this element
		 * @return T: The returned element
		 */
		T Get(int r, int c) const { return array[(r * col) + c]; }

		/**
		 * @brief Iterates through all the elements of
		 * the list, and executes the given function
		 *
		 * @param fun
		 */
		void Iterate(std::function<void(const T)> fun)
		{
			int i, j;
			for(i = 0; i < row; i++)
			{
				for(j = 0; j < col; j++)
				{
					fun(Get(i, j));
				}
			}
		}

		/**
		 * @brief Iterates through all the elements of
		 * the list, and executes the given function
		 *
		 * @param fun
		 */
		void Iterate(std::function<void(const T, const int x, const int y)> fun)
		{
			int i, j;
			for(i = 0; i < row; i++)
			{
				for(j = 0; j < col; j++)
				{
					fun(Get(i, j), i, j);
				}
			}
		}

		/**
		 * @brief Finds the item that satisfies the
		 * given condition and return the indices
		 * by passing reference of the indices
		 *
		 * @param fun The condition that is
		 * satisfied by the first element
		 * found
		 * @param fx the starting index of row.
		 * will be replaces by the location of the
		 * found item (by reference) if found, otherwise
		 * will contain -1
		 * @param fy the starting index of column.
		 * will be replaces by the location of the
		 * found item (by reference) if found, otherwise
		 * will contain -1
		 */
		void FindFirst(std::function<bool(const T item)> fun, int& fx, int& fy)
		{
			for(fx = 0; fx < row; fx++)
			{
				for(fy = 0; fy < col; fy++)
				{
					if(fun(Get(fx, fy)))
						return;
				}
			}
			fx = -1;
			fy = -1;
		}

		/**
		 * @brief Copy assignment operator
		 *
		 * @param other the object to copy from
		 * @return Array2D& The reference of the copied object
		 */
		Array2D& operator=(const Array2D<T>& other)
		{
			return *this = Array2D(other);
		}

		/**
		 * @brief Move assignment operator
		 *
		 * @param other the object to copy from
		 * @return Array2D& The reference of the newly created
		 * object that has its data moved from the other
		 */
		Array2D& operator=(Array2D<T>&& other)
		{
			row			= other.row;
			col			= other.col;
			array		= other.array;
			other.array = nullptr;
			other.row = other.col = 0;
		}

		/**
		 * @brief Checks if two Array2D objects are equivalent
		 * or not
		 *
		 * @param other the other array to compare with
		 * @return bool true if both arrays are
		 * equivalent, otherwise false
		 */
		bool operator==(const Array2D<T>& other)
		{
			if(row != other.row || col != other.col)
				return false;
			int i, j;
			for(i = 0; i < row; i++)
			{
				for(j = 0; j < col; j++)
				{
					if(Get(i, j) != other.Get(i, j))
						return false;
				}
			}
			return true;
		}

		/**
		 * @brief Checks if two Array2D objects are equivalent
		 * or not
		 *
		 * @param other the other array to compare with
		 * @return bool false if both arrays are
		 * equivalent, otherwise true
		 */
		bool operator!=(const Array2D<T>& other) { return !(*this == other); }

		/**
		 * @brief Destroy the Array2D object
		 *
		 */
		~Array2D()
		{
			if(array != nullptr)
				free(array);
		}
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, Array2D<T> array)
	{
		int row = array.GetRowCount(), col = array.GetColumnCount();
		if(row == 0 || col == 0)
		{
			os << "[ ]";
		}
		else
		{
			os << "[\n";
			int i, j;
			for(i = 0; i < row; i++)
			{
				os << "  [\t" << array.Get(i, 0);
				for(j = 1; j < col; j++)
				{
					os << ",\t" << array.Get(i, j);
				}
				os << " ]\n";
			}
			os << "]";
		}
		return os;
	}
} // namespace CustomArray

#pragma endregion Array_Region

#pragma region CustomList_Region

namespace List
{
	using namespace CustomArray;

	/**
	 * @brief Represents a single node in the list.
	 * Maintains a pointer of the data.
	 *
	 * @tparam T Type of the item stored by the list.
	 * Maintains T* within
	 */
	template <typename T> struct Node
	{
	  public:
		/**
		 * @brief Data within the node
		 *
		 */
		T* data;
		/**
		 * @brief The pointer to the neighbour fo this node
		 *
		 */
		Node *Next, *Prev;

		/**
		 * @brief Inserts this node before a given node
		 *
		 * @param other
		 */
		void InsertBefore(Node<T>* other)
		{
			Next		= other;
			Prev		= other->Prev;
			other->Prev = this;
			Prev->Next	= this;
		}
	};

	/**
	 * @brief A dynamic pointer list. Maintains the
	 * pointers, and if provieded, can delete itself and
	 * its children correctly
	 *
	 * @tparam T The type of the list, maintains T* within
	 */
	template <typename T> class PointerList
	{
	  private:
		/**
		 * @brief The head of the list
		 *
		 */
		Node<T>* Head;
		/**
		 * @brief The number of elements in the list
		 *
		 */
		int count = 0;

		/**
		 * @brief Create a Node object and
		 * increments the count. Only to be used internally
		 *
		 * @param data the data to store
		 * @return Node<T>* The reference to
		 * the newly created node is returned.
		 */
		Node<T>* CreateNode(T* data)
		{
			Node<T>* node = new Node<T>;
			node->data	  = data;
			node->Next = node->Prev = nullptr;
			return node;
		}

		/**
		 * @brief Adds a single node to the list
		 *
		 * @param nodeToAdd The new node to add to the list
		 * @param other The node to attach the new node on
		 */
		void AddNodeInList(Node<T>* nodeToAdd, Node<T>* other)
		{
			if(other == nullptr)
			{
				Head	   = nodeToAdd;
				Head->Next = Head->Prev = Head;
			}
			else
				nodeToAdd->InsertBefore(other);
			count++;
		}

		void DeleteNode(Node<T>* item)
		{
			if(count == 0)
				throw item;
			if(count == 1)
			{
				if(item != Head)
					throw item;
				Head  = nullptr;
				count = 0;
				return;
			}
			item->Next->Prev = item->Prev;
			item->Prev->Next = item->Next;
			if(Head == item)
				Head = item->Next;
			item->Next = item->Prev = nullptr;
			if(DeleteOnClear && item->data != nullptr)
				delete item->data;
			delete item;
			count--;
		}

	  public:
		/**
		 * @brief If true, when the `Clear()` function is
		 * called (either by the destructor, or explicitly),
		 * all pointers maintained by the list
		 * is also deleted.
		 *
		 */
		bool DeleteOnClear;

		/**
		 * @brief Construct a new Pointer List object
		 *
		 * @param delOnClear : If true, then all items are
		 * deleted when `Clear()` function is called
		 * or when this destructor is called
		 */
		PointerList(bool delOnClear = true)
		{
			count		  = 0;
			Head		  = nullptr;
			DeleteOnClear = delOnClear;
		}

		/**
		 * @brief Construct a new Pointer List object
		 *
		 * Copy contructor to obey the rule of 5 in C++
		 * @param list reference of the list to copy
		 */
		PointerList(PointerList<T>& list)
		{
			if(list.count == 0)
			{
				Head = nullptr;
			}
			else
			{
				Node<T>* x = list.Head;
				for(int i = 0; i < list.count; i++)
				{
					T* data = new T(*(x->data));
					Add(data);
					x = x->Next;
				}
			}
			count		  = list.count;
			DeleteOnClear = list.DeleteOnClear;
		}

		/**
		 * @brief Move Constructor for Pointer List object.
		 * Declared to obey the rule of 5 in C++
		 * @param list
		 */
		PointerList(PointerList<T>&& list)
		{
			count			   = list.count;
			Head			   = list.Head;
			DeleteOnClear	   = list.DeleteOnClear;
			list.count		   = 0;
			list.Head		   = nullptr;
			list.DeleteOnClear = true;
		}

		/**
		 * @brief Adds a pointer to the list
		 *
		 * @param data The data to add to the list
		 * @param end If true, item is appended at the
		 * end of the list. Otherwise it is
		 * inserted at the beginning of the list.
		 */
		void Add(T* data, bool end = true)
		{
			Node<T>* x = CreateNode(data);
			AddNodeInList(x, Head);
		}

		/**
		 * @brief Adds a pointer to the list
		 *
		 * @param data The data to add to the list
		 * @param end If true, item is appended at the end
		 * of the list. Otherwise it is inserted at the
		 * beginning of the list.
		 */
		void Add(const T data, bool end = true)
		{
			T* x = new T;
			*x	 = data;
			Add(x);
			if(end == false)
				Head = Head->Prev;
		}

		/**
		 * @brief Adds the elements of the initializer
		 * to this list
		 *
		 * @param list the initializer list whose elements
		 * are to be copied in this list
		 * @param end If true, elements are appended
		 * at the end of list. Otherwise they are
		 * added to the beginning of the list.
		 */
		void Add(std::initializer_list<T> list, bool end = true)
		{
			for(const T x: list)
			{
				Add(x, end);
			}
		}

		/**
		 * @brief Adds the elements of the array
		 * to the list
		 *
		 * @param array The array whose elements
		 * are to be copied to this list
		 * @param end If true, elements are appended
		 * at the end of list. Otherwise they are
		 * added to the beginning of the list.
		 */
		void Add(Array1D<T> array, bool end = true)
		{
			for(int i = 0; i < array.GetCount(); i++)
			{
				Add(array.Get(i), end);
			}
		}

		/**
		 * @brief Get the number of elements in this
		 * list
		 *
		 * @return int
		 */
		int GetCount() { return count; }

		/**
		 * @brief Removes the first element of the list. The
		 * pointer contained within it is returned.
		 *
		 * @return T* The pointer contained within. If
		 * deletion fails, nullptr is returned.
		 */
		T* RemoveFirst()
		{
			if(Head == nullptr)
				return nullptr;
			T* data	   = Head->data;
			Head->data = nullptr;
			Node<T>* x = Head;
			if(count == 1)
			{
				Head = nullptr;
			}
			else
			{
				Head->Next->Prev = Head->Prev;
				Head->Prev->Next = Head->Next;
				Head			 = Head->Next;
			}
			delete x;
			return data;
		}

		/**
		 * @brief Removes the first element of the list, and
		 * deletes the pointer item as well
		 *
		 * @return bool: If true, deletion is successful.
		 * Otherwise it is not successful
		 */
		bool RemoveFirstAndDelete()
		{
			if(count == 0)
				return false;
			T* x = RemoveFirst();
			if(x != nullptr)
				delete x;
			return true;
		}

		/**
		 * @brief Removes the last element of the list. The
		 * pointer contained within it is returned.
		 *
		 * @return T* The pointer contained within. If
		 * deletion fails, nullptr is returned.
		 */
		T* RemoveLast()
		{
			if(count < 2)
				return RemoveFirst();
			Node<T>* x	  = Head->Prev;
			T*		 data = x->data;
			x->data		  = nullptr;
			x->Next->Prev = x->Prev;
			x->Prev->Next = x->Next;
			delete x;
			return data;
		}

		/**
		 * @brief Removes the last element of the list,
		 * and deletes the pointer item as well
		 *
		 * @return bool: If true, deletion is successful.
		 * Otherwise it is not successful
		 */
		bool RemoveLastAndDelete()
		{
			T* x = RemoveLast();
			if(x == nullptr)
				return false;
			delete x;
			return true;
		}

		/**
		 * @brief Iterates along all the elements of the
		 * list with the given function
		 *
		 * @param fun The function object provided
		 * for the iteration.
		 */
		void Iterate(std::function<void(const T)> fun)
		{
			if(count == 0)
				return;
			Node<T>* node = Head;
			do
			{
				fun(*(node->data));
				node = node->Next;
			} while(node != Head);
		}

		/**
		 * @brief Deletes all elements with the
		 * list that satisfy the given condition
		 *
		 * @param condition The function object that
		 * checks for the satisfaction of a condition
		 * for a given element.
		 *
		 * @return int The number of elements deleted.
		 */
		void DeleteAll(std::function<bool(const T)> condition)
		{
			if(count == 0)
				return;
			int		 times = count;
			Node<T>* node  = Head;
			do
			{
				if(condition(*(node->data)))
				{
					Node<T>* x = node;
					node	   = node->Next;
					DeleteNode(x);
				}
				else
					node = node->Next;
			} while(--times > 0);
		}

		/**
		 * @brief Clears the list. If `DeleteOnClear`
		 * is true, the pointer elements are deleted as well.
		 *
		 * @return int The number of elements deleted.
		 */
		int Clear()
		{
			int result = count;
			if(result == 0)
				return count;
			count			 = 0;
			Head->Prev->Next = nullptr;
			do
			{
				if(DeleteOnClear)
					delete Head->data;
				else
					Head->data = nullptr;
				Node<T>* x = Head;
				Head	   = Head->Next;
				delete x;
			} while(Head != nullptr);
			return result;
		}

		/**
		 * @brief Converts the list into a
		 * Array object of type T* and size `count`.
		 * This array is allocated using `malloc()`,
		 * and therefore must be freed using `free()`
		 *
		 * @return T*
		 */
		T* GetArray()
		{
			if(count == 0)
				return nullptr;
			T*		 array = MALLOC<T>(count);
			int		 pos   = 0;
			Node<T>* x	   = Head;
			do
			{
				T data		 = *(x->data);
				array[pos++] = data;
				x			 = x->Next;
			} while(Head != x);
			return array;
		}

		/**
		 * @brief Converts the list into a
		 * Array1D object of type
		 * Array1D<T> and size `count`.
		 *
		 * @return Array::Array1D<T>
		 */
		Array1D<T> GetArray1D()
		{
			Array1D<T> arr(count);
			Node<T>*   x = Head;
			for(int i = 0; i < count; i++)
			{
				arr.Set(i, *x->data);
				x = x->Next;
			}
			return arr;
		}

		/**
		 * @brief Checks the contents of two list and sees if
		 * they are equal or not
		 *
		 * @param a List to check
		 * @param b List to check
		 * @return bool True if the lists are copies of each other,
		 * otherwise False.
		 */
		static bool Equal(PointerList<T> a, PointerList<T> b)
		{
			if(a.Head == b.Head)
				return a.count == b.count;
			else if(a.count != b.count)
				return false;
			int x = a.count;
			if(x == 0)
				return true;
			Node<T>*an = a.Head, *bn = b.Head;
			do
			{
				if(an->data != bn->data)
					return false;
				an = an->Next;
				bn = bn->Next;
			} while(an != a.Head && bn != b.Head);
			return true;
		}

		PointerList& operator=(const PointerList<T>& otherlist)
		{
			return *this = PointerList(otherlist);
		}

		/**
		 * @brief Move assignment operator.
		 * Defined to obey the rule of 5 in C++
		 *
		 * @param otherlist the list to move from
		 * @return PointerList& The moved list reference
		 */
		PointerList& operator=(PointerList<T>&& otherlist)
		{
			count					= otherlist.count;
			Head					= otherlist.Head;
			DeleteOnClear			= otherlist.DeleteOnClear;
			otherlist.count			= 0;
			otherlist.Head			= nullptr;
			otherlist.DeleteOnClear = true;
			return *this;
		}

		/**
		 * @brief Destroy the Pointer List object
		 *
		 */
		~PointerList() { Clear(); }
	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, PointerList<T>& list)
	{
		if(list.GetCount() == 0)
		{
			os << "[ ]";
			return os;
		}
		os << "[ ";
		list.Iterate([&os](const T item) -> void { os << item << ", "; });
		os << "]";
		return os;
	}
} // namespace List

namespace CustomArray
{
	/**
	 * @brief Represents a read-only Compressed Sparse Matrix object
	 *
	 * @tparam T
	 */
	template <typename T> class CSR_Matrix
	{
	  public:
		/**
		 * @brief Represents the dimension of this
		 * Compressed Sparse Row(CSR) Matrix object
		 *
		 */
		Vertex row, col;
		/**
		 * @brief The collection of non-zero elements
		 * in this matrix
		 *
		 */
		T* NNZ;
		/**
		 * @brief The indices of the column elements
		 * of the matrix which are at the same indices
		 * as the NNZ elements
		 *
		 */
		Vertex* col_index;
		/**
		 * @brief The compressed row indices of all
		 * elements in the matrix
		 *
		 */
		Vertex* row_index;

		/**
		 * @brief Construct a new CSR_Matrix object
		 * from a 2D array. Only creates a copy
		 * of the non-zero elements
		 *
		 * @param matrix : The input 2D matrix
		 * @param lowerTriangleOnly : Copies only the
		 * lower trinagular part of the matrix
		 */
		CSR_Matrix(Array2D<T>& matrix, bool lowerTriangleOnly = false)
		{
			T					 zero = T();
			List::PointerList<T> NNZ_list(true);
			Vertex				 i, j;
			row = matrix.GetRowCount();
			col = matrix.GetColumnCount();
			List::PointerList<Vertex> col_list(true), row_list(true);
			Vertex					  z = 0;
			row_list.Add(z);
			for(i = 0; i < row; i++)
			{
				for(j = 0; j < col; j++)
				{
					if(lowerTriangleOnly && j <= i)
						continue;
					T item = matrix.Get(i, j);
					if(item != zero)
					{
						NNZ_list.Add(item);
						col_list.Add(j);
					}
				}
				row_list.Add(col_list.GetCount());
			}
			NNZ		  = NNZ_list.GetArray();
			col_index = col_list.GetArray();
			row_index = row_list.GetArray();
		}

		/**
		 * @brief Construct a new csr matrix object
		 * from the given raw arrays. The raw arrays must be
		 * allocated using `malloc()`, since the destructor
		 * calls free() on them
		 *
		 * @param nnz The array of non-zero elements
		 * @param row_idxs The cumulative count of row elements per row
		 * @param rows The number of rows
		 * @param col_idxs The column-indices of each element
		 * @param cols The number of columns
		 * @param copy If true, a copy of these elements are made, otherwise 
		 * they are referenced. In this case, the arrays must be allocated using
		 * `malloc()`
		 */
		CSR_Matrix(T* nnz, Vertex* row_idxs, Vertex rows, Vertex* col_idxs,
				   Vertex cols, bool copy)
		{
			row = rows;
			col = cols;
			if(rows == 0 || cols == 0 || nnz == nullptr ||
			   row_idxs == nullptr || col_idxs == nullptr)
			{
				row_index = col_index = nullptr;
				NNZ					  = nullptr;
			}
			else if(copy == false)
			{
				row_index = row_idxs;
				col_index = col_idxs;
				NNZ		  = nnz;
			}
			else
			{
				Vertex count = row_idxs[row];
				row_index	 = MALLOC<Vertex>(row + 1);
				NNZ			 = MALLOC<T>(count);
				col_index	 = MALLOC<Vertex>(count);
				std::copy(nnz, nnz + count, NNZ);
				std::copy(col_idxs, col_idxs + count, col_index);
				std::copy(row_idxs, row_idxs + row + 1, row_index);
			}
		}

		/**
		 * @brief Copy constructor for CSR_Matrix.
		 * Implemented to obey rule of 5
		 *
		 * @param other the object to copy from
		 */
		CSR_Matrix(const CSR_Matrix<T>& other)
		{
			if(other.row <= 0 || other.col <= 0)
			{
				row = col = 0;
				NNZ		  = nullptr;
				col_index = nullptr;
				row_index = MALLOC<Vertex>(1);
			}
			else
			{
				row		  = other.row;
				col		  = other.col;
				row_index = MALLOC<Vertex>(row + 1);
				Vertex nnz;
				std::copy(other.row_index, other.row_index + row + 1,
						  row_index);
				nnz		  = row_index[row];
				NNZ		  = MALLOC<T>(nnz);
				col_index = MALLOC<Vertex>(nnz);
				std::copy(other.NNZ, other.NNZ + nnz, NNZ);
				std::copy(other.col_index, other.col_index + nnz, col_index);
			}
		}

		/**
		 * @brief Move Construct for CSR_Matrix.
		 * Implemented to obey rule of 5
		 *
		 * @param other the object to move from
		 */
		CSR_Matrix(CSR_Matrix<T>&& other)
		{
			row				= other.row;
			col				= other.col;
			row_index		= other.row_index;
			col_index		= other.col_index;
			NNZ				= other.NNZ;
			other.row_index = nullptr;
			other.col_index = nullptr;
			other.NNZ		= nullptr;
			row = col = 0;
		}

		/**
		 * @brief Get the Row Count
		 *
		 * @return int Rows in this matrix
		 */
		Vertex GetRowCount() { return row; }

		/**
		 * @brief Get the Column Count
		 *
		 * @return int Columns in this matrix
		 */
		Vertex GetColumnCount() { return col; }

		/**
		 * @brief Get the Number of non-zero elements
		 *
		 * @return int Number of non-zero elements
		 */
		Vertex Non_Zero_Elements_Count() { return row_index[row]; }

		/**
		 * @brief Iterates through the elements in
		 * this matrix
		 *
		 * @param fn The function to perform on
		 * each element
		 * @param only_non_zero_elements If true,
		 * function is called only on the non-zero
		 * elements. Otherwise it is called on all
		 * elements of the matrix.
		 */
		void Iterate(std::function<void(const T item)> fn,
					 bool only_non_zero_elements = true)
		{
			if(only_non_zero_elements)
			{
				Vertex size = Non_Zero_Elements_Count();
				for(Vertex i = 0; i < size; i++)
				{
					fn(NNZ[i]);
				}
			}
			else
			{
				const T zero = T();
				int		i, j, k, p = 0;
				for(i = 0; i < row; i++)
				{
					k = row_index[i + 1] - row_index[i];
					for(j = 0; j < col; j++)
					{
						if(k > 0 && j == col_index[p])
						{
							fn(NNZ[p++]);
							k--;
						}
						else
							fn(zero);
					}
				}
			}
		}

		/**
		 * @brief Iterates through the elements in
		 * this matrix, with their correspoinding
		 * indices
		 *
		 * @param fn The function to perform on
		 * each element
		 * @param only_non_zero_elements If true,
		 * function is called only on the non-zero
		 * elements. Otherwise it is called on all
		 * elements of the matrix.
		 */
		void Iterate(std::function<void(const T item, Vertex x, Vertex y)> fn,
					 bool only_non_zero_elements = true)
		{
			if(only_non_zero_elements)
			{
				Vertex len = Non_Zero_Elements_Count();
				if(len == 0)
					return;
				Vertex x = 0, xr = row_index[1];
				for(Vertex i = 0; i < len; i++)
				{
					Vertex y = col_index[i];
					while(xr == 0)
					{
						x++;
						xr = row_index[x + 1] - row_index[x];
					}
					fn(NNZ[i], x, y);
					xr--;
				}
			}
			else
			{
				const T zero = T();
				Vertex	i, j, k, p = 0;
				for(i = 0; i < row; i++)
				{
					k = row_index[i + 1] - row_index[i];
					for(j = 0; j < col; j++)
					{
						if(k > 0 && j == col_index[p])
						{
							fn(NNZ[p++], i, j);
							k--;
						}
						else
							fn(zero, i, j);
					}
				}
			}
		}

		/**
		 * @brief Gets the elements at this position of the matrix
		 *
		 * @param x The queried row
		 * @param y The queried column
		 * @return T The value stored in this position
		 */
		T Get(Vertex x, Vertex y)
		{
			if(x >= row || y >= col)
				throw x + y;
			Vertex s  = row_index[x];	  // end of row element
			Vertex e  = row_index[x + 1]; // start of row element
			Vertex rc = e - s;			  // number of elements in the row
			if(rc <= 0)
				return T();				  // Zero
			for(Vertex i = s; i < e; i++) // search for element
				if(col_index[i] == y)
					return NNZ[i]; // found element
			return T();			   // Zero
		}

		/**
		 * @brief Return a Array2D
		 * for the data contained
		 *
		 * @return Array2D<T>
		 */
		Array2D<T> GetArray()
		{
			Array2D<T> created(row, col); // Calling the constructor to set all elements to 0
			for(int i=0;i<row;i++)
			{
				for(int j=0;j<col;j++)
				{
					created.Set(i, j, 0);
				}
			}
			Iterate(
				[&created](const T item, const Vertex x, const Vertex y) -> void
				{ created.Set(x, y, item); },
				true); // Copy the non-zero elements
			return created;
		}

		/**
		 * @brief Copy assignment operator.
		 * Implemented to obey rule of 5
		 *
		 * @param other The object to copy
		 * @return CSR_Matrix& The returned
		 * reference
		 */
		CSR_Matrix& operator=(const CSR_Matrix<T>& other)
		{
			return *this = CSR_Matrix(other);
		}

		/**
		 * @brief Move assignment operator.
		 * Implemented to obey rule of 5
		 *
		 * @param other The object to copy
		 * @return CSR_Matrix& the returned
		 * reference
		 */
		CSR_Matrix& operator=(CSR_Matrix<T>&& other)
		{
			row				= other.row;
			col				= other.col;
			row_index		= other.row_index;
			col_index		= other.col_index;
			NNZ				= other.NNZ;
			other.row_index = nullptr;
			other.col_index = nullptr;
			other.NNZ		= nullptr;
			row = col = 0;
			return *this;
		}

		/**
		 * @brief Destroy the CSR_Matrix object
		 *
		 */
		~CSR_Matrix()
		{
			if(col_index != nullptr)
				free(col_index);
			if(row_index != nullptr)
				free(row_index);
			if(NNZ != nullptr)
				free(NNZ);
		}
	};
} // namespace CustomArray

#pragma endregion CustomList_Region

#pragma region Fib_Heap

namespace Heap
{
	/**
	 * @brief Represents a node in the Fibonacci Heap
	 *
	 */
	template <typename K, typename T> struct FibonacciNode
	{
		/**
		 * @brief Pointers
		 *
		 */
		FibonacciNode *Parent, *LeftMostChild, *NextSibiling, *PrevSibiling;
		/**
		 * @brief Data fields
		 *
		 */
		int Degree;
		T	Data;
		K	Key;
		/**
		 * @brief Data fields
		 *
		 */
		bool Marked;
	};

	/**
	 * @brief Create a Node object
	 *
	 * @param data: The data contained by this node
	 * @return FibonacciNode* Created node
	 */
	template <typename K, typename T>
	FibonacciNode<K, T>* CreateNode(K key, T data)
	{
		FibonacciNode<K, T>* node = new FibonacciNode<K, T>;
		node->Data				  = data;
		node->Key				  = key;
		node->Parent = node->LeftMostChild = node->NextSibiling =
			node->PrevSibiling			   = nullptr;
		node->Degree					   = 0;
		node->Marked					   = false;
		return node;
	}

	/**
	 * @brief Joins/connects the two circular list of nodes together
	 *
	 * @param n1 Circular list of sibiling nodes
	 * @param n2 Circular list of sibiling nodes
	 */
	template <typename K, typename T>
	void Merge_Fibonacci_Nodes(FibonacciNode<K, T>* n1h,
							   FibonacciNode<K, T>* n2h)
	{
		FibonacciNode<K, T>*n1p = n1h->PrevSibiling, *n2p = n2h->PrevSibiling;
		n1p->NextSibiling = n2h;
		n2h->PrevSibiling = n1p;
		n2p->NextSibiling = n1h;
		n1h->PrevSibiling = n2p;
	}

	/**
	 * @brief De-links this node with its sibilings
	 *
	 * @param node The node to de-link
	 */
	template <typename K, typename T>
	void Unattach_Node(FibonacciNode<K, T>* node)
	{
		if(node->NextSibiling != nullptr && node->PrevSibiling != nullptr)
		{
			node->PrevSibiling->NextSibiling = node->NextSibiling;
			node->NextSibiling->PrevSibiling = node->PrevSibiling;
		}
	}

	/**
	 * @brief Attaches one node as the child of the other
	 *
	 * @param parent The node which is to be the parent
	 * @param child The node which is to be the child
	 */
	template <typename K, typename T>
	void Add_Child(FibonacciNode<K, T>* parent, FibonacciNode<K, T>* child)
	{
		if(parent->Degree == 0 || parent->LeftMostChild == nullptr)
		{
			parent->Degree		  = 0;
			parent->LeftMostChild = child;
			child->PrevSibiling = child->NextSibiling = child;
		}
		else
		{
			child->NextSibiling = parent->LeftMostChild;
			child->PrevSibiling = parent->LeftMostChild->PrevSibiling;
			child->NextSibiling->PrevSibiling = child;
			child->PrevSibiling->NextSibiling = child;
		}
		child->Marked = false;
		parent->Degree++;
		child->Parent = parent;
	}

	int Log2(unsigned int x)
	{
		int bits = 0;
		while(x > 0)
		{
			bits++;
			x >>= 1;
		}
		return bits;
	}

	template <typename K, typename T> class FibonacciHeap
	{
	  private:
		FibonacciNode<K, T>* Root;
		int					 Count;

		void Consolidate()
		{
			FibonacciNode<K, T>* node = Root;
			Root					  = nullptr;
			int					  i, d, max_degree = Log2(Count) + 1;
			FibonacciNode<K, T>** array = MALLOC<FibonacciNode<K, T>*>(max_degree);//new FibonacciNode<K, T>*[max_degree];
			FibonacciNode<K, T>*  next, *w;
			for(i = 0; i < max_degree; i++)
				array[i] = nullptr;
			node->PrevSibiling->NextSibiling = nullptr;
			do
			{
				next = node->NextSibiling;
				if(next != nullptr)
					node->NextSibiling = node->NextSibiling->PrevSibiling =
						nullptr;
				node->PrevSibiling = nullptr;
				d				   = node->Degree;
				while(array[d] != nullptr)
				{
					w = array[d];
					if(w->Data < node->Data)
					{
						Add_Child(w, node);
						node = w;
					}
					else
						Add_Child(node, w);
					array[d] = nullptr;
					d		 = node->Degree;
				}
				array[d] = node;
				node	 = next;
			} while(node != nullptr);
			for(i = 0; i < max_degree; i++)
			{
				if(array[i] != nullptr)
					Insert(array[i]);
			}
			free(array);
		}

		void Cut(FibonacciNode<K, T>* x, FibonacciNode<K, T>* y)
		{
			if(x->NextSibiling == x)
				y->LeftMostChild = nullptr;
			else
			{
				y->LeftMostChild = x->NextSibiling;
				Unattach_Node(x);
			}
			y->Degree--;
			x->Parent = nullptr;
			x->Marked = false;
			Insert(x); // Also ensures x becomes root if it is the minimum
		}

		void CascadingCut(FibonacciNode<K, T>* x)
		{
			FibonacciNode<K, T>* y = x->Parent;
			if(y != nullptr)
			{
				if(!(x->Marked))
					x->Marked = true;
				else
				{
					Cut(x, y);
					CascadingCut(y);
				}
			}
		}

		void Insert(FibonacciNode<K, T>* node)
		{
			if(Root == nullptr)
			{
				Root			   = node;
				Root->NextSibiling = Root;
				Root->PrevSibiling = Root;
			}
			else
			{
				node->NextSibiling				 = Root;
				node->PrevSibiling				 = Root->PrevSibiling;
				Root->PrevSibiling->NextSibiling = node;
				Root->PrevSibiling				 = node;
				if(Root->Data > node->Data)
				{
					Root = node;
				}
			}
		}

	  public:
		/**
		 * @brief Construct a new Fibonacci Heap object
		 *
		 */
		FibonacciHeap()
		{
			Count = 0;
			Root  = nullptr;
		}

		/**
		 * @brief Destroy the Fibonacci Heap object
		 *
		 */
		~FibonacciHeap()
		{
			while(Count > 0)
				DeleteMin();
		}

		/**
		 * @brief Get the minimum value of the heap
		 *
		 * @return int
		 */
		T GetTop() { return Root->Data; }

		K GetTopKey() { return Root->Key; }

		/**
		 * @brief Inserts a value to the heap
		 *
		 * @param value
		 */
		FibonacciNode<K, T>* Insert(K object, T value)
		{
			FibonacciNode<K, T>* node = CreateNode<K, T>(object, value);
			Insert(node);
			Count++;
			return node;
		}

		/**
		 * @brief Joins another Heap with this heap
		 *
		 * @param otherTree The heap to join
		 */
		void Union(FibonacciHeap<K, T>& otherTree)
		{
			if(otherTree.Count == 0)
				return;
			if(Count == 0)
				Root = otherTree.Root;
			else
				Merge_Fibonacci_Nodes(Root, otherTree.Root);
			Count += otherTree.Count;
			otherTree.Root	= nullptr;
			otherTree.Count = 0;
		}

		/**
		 * @brief Removes the least value of the heap
		 *
		 * @return true Deletion successful
		 * @return false Deletion failed as there was no node to delete
		 */
		bool DeleteMin()
		{
			if(Count == 0)
				return false;
			else if(Count == 1)
			{
				delete Root;
				Root = nullptr;
			}
			else
			{
				FibonacciNode<K, T>* c = Root->LeftMostChild;
				if(c != nullptr)
				{
					FibonacciNode<K, T>* x = c;
					do
					{
						x->Parent = nullptr;
						x		  = x->NextSibiling;
					} while(x != c);
					Merge_Fibonacci_Nodes(Root, c);
				}
				c	 = Root;
				Root = c->NextSibiling;
				// This is temporary, Root is set again in Consolidate()
				Unattach_Node(c);
				Consolidate();
				delete c;
			}
			Count--;
			return true;
		}

		/**
		 * @brief Decreses the value of a node existing in the heap
		 *
		 * @param node The node whose value is to be reduced
		 * @param newvalue The new value replaced in the node
		 * @return true Process completed successfully
		 * @return false newvalue is greater than the value of the node
		 */
		bool DecreaseKey(FibonacciNode<K, T>* node, T newvalue)
		{
			if(node->Data < newvalue)
				return false;
			node->Data			   = newvalue;
			FibonacciNode<K, T>* p = node->Parent;
			if(p != nullptr && node->Data < p->Data)
			{
				Cut(node, p);
				CascadingCut(p);
			} // Cut() also ensures node becomes the root if it is minimum
			if(newvalue < Root->Data)
			{
				Root = node;
			}
			return true;
		}

		/**
		 * @brief Get the number of elements in the heap
		 *
		 * @return int
		 */
		int GetSize() { return Count; }

		void Clear()
		{
			while(Count > 0)
				DeleteMin();
		}
	};
} // namespace Heap

#pragma endregion Fib_Heap

#pragma region Graph

namespace Graphs
{
	/**
	 * @brief The Abstract Data structure
	 * of a graph.
	 * Every Graph Data structure in this
	 * project must implement this class.
	 * simple graph, weighted and Directed/undirected
	 * are supported.
	 *
	 *
	 * The nodes of the graph is fixed to be a single
	 * byte integer (unsigned char), and its weights
	 * are unsigned integers. the
	 *
	 */
	class graphADT
	{
	  public:
		/**
		 * @brief This function checks
		 * if the graph is a directed
		 * graph or not
		 *
		 * @return bool: True if
		 * Graph is directed, otherwise
		 * undirected
		 */
		virtual bool isDirected() = 0;
		/**
		 * @brief Get the number of Vertices `V`
		 * in this graph. This graph will contain
		 * vertices 0, 1, 2, ... V-1
		 *
		 * @return int Number of vertices
		 */
		virtual Vertex get_Vertices() = 0;
		/**
		 * @brief Get the Edge Count for
		 * the graph
		 *
		 * @return Vertex: The number of edges of the graph
		 */
		virtual Vertex get_Edge_Count() = 0;
		/**
		 * @brief Get the Weight between
		 * two vertices u and v
		 *
		 * @param u A vertex in this graph
		 * @param v A vertex in this graph
		 * @return uint The weight of the edge
		 */
		virtual Weight get_Weight(Vertex u, Vertex v) = 0;
		/**
		 * @brief Add an vertex in the graph
		 *
		 * @param v: The vertex to add to the graph
		 * @return bool : true means that vertex addition was successful,
		 * otherwise vertex addition failed/not supported
		 */
		virtual bool add_Vertex(Vertex v) = 0;
		/**
		 * @brief Removes a vertex (and any associated edges)
		 * from the graph
		 *
		 * @param v the vertex to remove
		 * @return bool: true means that the vertex removal was successful,
		 * otherwise vertex removal failed/not supported
		 */
		virtual bool remove_Vertex(Vertex v) = 0;
		/**
		 * @brief Updates an edge of the graph
		 *
		 * @param u the vertex from where the edge begins
		 * @param v the vertex on which the edge ends
		 * @param w the weight of the edge. If 0, the edge is set for removal,
		 * otherwise the edge is set to be added to the graph
		 *
		 * @return bool: true means that the edge updation was successful,
		 * otherwise edge updation failed/not supported
		 */
		virtual bool update_Edge(Vertex u, Vertex v, Weight w) = 0;
		/**
		 * @brief Iterates all the edges with their neighbouring vertices
		 * for a given vertex in the graph
		 *
		 * @param v The vertex whose neighbours are to be iterated
		 * @param fn The function to execute on each neighbour
		 * @param mode The mode of iteration. 1 for only neighbours where edge
		 * is incoming to v  (that is, all edges (x,v), where v is the queried
		 * vertex), 2 for only neighbours to which edge is outgoing from v (that
		 * is, all edges (v,x), where v is * the queried vertex), and 3 for
		 * both. In mode 3, the nodes that are both incoming and outgoing
		 * neighbours get iterated twice.
		 */
		virtual void iterate_Neighbours(
			Vertex v,
			std::function<void(const Vertex u, const Vertex v, const Weight w)>
				fn,
			int mode = 3) = 0;
		/**
		 * @brief Iterates all the edges within the graph
		 *
		 * @param fn The function to execute on each edge
		 */
		virtual void iterate_Edges(
			std::function<void(const Vertex u, const Vertex v, const Weight w)>
				fn) = 0;
	};

	/**
	 * @brief Represents a single weighted edge of
	 * a simple graph
	 *
	 */
	struct GraphEdge
	{
	  public:
		/**
		 * @brief The vertices involved
		 * in the edge
		 *
		 */
		Vertex u, v;
		/**
		 * @brief The weight of the edge
		 *
		 */
		Weight w;

		GraphEdge(Vertex _u, Vertex _v, Weight _w)
		{
			u = _u;
			v = _v;
			w = _w;
		}

		GraphEdge()
		{
			u = 0;
			v = 0;
			w = 0;
		}

		bool operator<(const GraphEdge edge) const
		{
			if(u != edge.u)
				return u < edge.u;
			else
				return v < edge.v;
		}
	};

	#pragma region SparseGraphComponents

	/**
	 * @brief A sparse graph is a graph which stores its
	 * adjacency matrix in CSR (Compressed Sparse Row) form,
	 * and uses the CSR_Matrix internally for this purpose
	 *
	 */
	class SparseGraph : public graphADT
	{
		void PrepareMatrix(CustomArray::Array1D<GraphEdge>& edges,
						   const Vertex vertices, const bool directed,
						   bool overwrite = false)
		{
			Vertex elements;
			edges.Sort();
			if(directed)
				elements = edges.GetCount();
			else
			{
				elements = 0;
				edges.Iterate(
					[&elements](const GraphEdge e) -> void
					{
						if(e.u < e.v)
							elements++;
					});
			}
			Vertex							   i, edge_count = 0;
			std::unordered_map<Weight, Weight> index_check;
			for(i = 0; i < edges.GetCount(); i++)
			{
				Weight hash;
				auto   edg = edges.Get(i);
				if(edg.u > edg.v && directed)
				{
					hash = edg.v;
					hash = (hash << 32) + edg.u;
				}
				else
				{
					hash = edg.u;
					hash = (hash << 32) + edg.v;
				}
				if(index_check.count(hash) != 0)
				{
					if(index_check[hash] == edg.w)
					{
						#if DEBUG
						std::cout << "Warning! Duplicate key exists!\nFound ("
								  << edg.u << ", " << edg.v << "with weight "
								  << edg.w << "\n";
						#endif
						continue;
					}
					#if DEBUG
					std::cout << "Duplicate key exists! Graph edgelist is "
								 "invalid!\nFound ("
							  << edg.u << ", " << edg.v << "with weights "
							  << edg.w << " and " << index_check[hash] << "\n";
					#endif
					throw hash;
				}
				else if(edg.u == edg.v)
				{
					#if DEBUG
					std::cout << "Warning! Self-edge detected! for (" << edg.u
							  << ", " << edg.v << ")\n\n";
					#endif
					continue;
				}
				index_check[hash] = edg.w;
				edge_count++;
			}
			if(!overwrite)
			{
				AdjMat.NNZ		 = new Weight[elements];
				AdjMat.col_index = new Vertex[elements];
				AdjMat.row_index = new Vertex[vertices + 1];
			}
			Weight* nnz	  = AdjMat.NNZ;
			Vertex *c_idx = AdjMat.col_index, *r_idx = AdjMat.row_index;
			AdjMat.col = AdjMat.row = vertices;
			bool invalid			= false;
			for(i = 0; i <= vertices; i++)
				r_idx[i] = 0;
			i = 0;
			edges.Iterate(
				[&](const GraphEdge e) -> void
				{
					if(e.u >= vertices || e.v >= vertices)
						invalid = true;
					else if(directed || e.u < e.v)
					{
						nnz[i]	 = e.w;
						c_idx[i] = e.v;
						r_idx[1 + e.u]++;
						i++;
					}
				});
			if(invalid)
			{
				#if DEBUG
				std::cout << "Invalid edge(s)! edgelist is invalid!\n\n";
				#endif
				throw edges;
			}
			for(i = 1; i <= vertices; i++)
				r_idx[i] += r_idx[i - 1];
			VertexCount = vertices;
			IsDirected	= directed;
		}

	  public:
		Vertex							VertexCount;
		bool							IsDirected;
		CustomArray::CSR_Matrix<Weight> AdjMat;

		/**
		 * @brief This overwrites a new edgelist into the current graph without
		 * any new allocations. It can not expand/increase the list of edges.
		 * Only valid for specific cases, do not use this unless you know what
		 * you are doing :-)
		 *
		 * @param new_edges The array of new edges, that will overwrite all old
		 * edges
		 * @param new_vertices The number of vertices for the new edgelist
		 * @return true if overwriting is successful. false otherwise
		 */
		bool Unsafe_Overwrite(CustomArray::Array1D<GraphEdge>& new_edges,
							  const Vertex					   new_vertices)
		{
			size_t new_size = CSR_SIZE(new_edges.GetCount(), new_vertices);
			size_t cur_size =
				CSR_SIZE(VertexCount, AdjMat.Non_Zero_Elements_Count());
			if(new_size > cur_size)
				return false;
			PrepareMatrix(new_edges, new_vertices, false, true);
			return true;
		}

		/**
		 * @brief Construct a new Sparse Graph object
		 * from a given 2D matrix representation
		 * of the adjacency matrix of the graph
		 *
		 * @param edges The adjacency matrix of the graph
		 * @param vertices The number of vertices, or the
		 * width of the square matrix `edges`
		 * @param directed if true, only the lower triangle
		 * of the adjacency matrix is considered, that is
		 * the edges (u, v) where u <= v
		 */
		SparseGraph(CustomArray::Array1D<GraphEdge>& edges, Vertex vertices,
					bool directed)
			: AdjMat(nullptr, nullptr, 0, nullptr, 0, true)
		{
			if(vertices < 2)
				throw vertices; // Trivial graph of only one element
			VertexCount = vertices;
			IsDirected	= directed;
			PrepareMatrix(edges, vertices, directed);
		}

		/**
		 * @brief Construct a new Sparse Graph object
		 *
		 * @param edges The 2D matrix representation of
		 * the adjacency matrix
		 * @param vertices The number of vertices, or the
		 * width of the square matrix `edges`
		 * @param directed if true, only the lower triangle
		 * of the adjacency matrix is considered.
		 */
		SparseGraph(CustomArray::Array2D<Weight>& edges, Vertex vertices,
					bool directed)
			: AdjMat(edges, !directed)
		{
			if(vertices < 2)
				throw vertices; // Trivial graph of only one element
			else if(edges.GetRowCount() != edges.GetColumnCount() ||
					(Vertex)edges.GetColumnCount() != vertices)
				throw edges; // Invalid format of matrix
			VertexCount = vertices;
			IsDirected	= directed;
		}

		/**
		 * @brief Construct a new Sparse Graph object
		 *
		 * @param edges CSR_Matrix of edges
		 * @param vertices Number of vertices in the graph
		 * @param directed Whether or not the graph is directed
		 */
		SparseGraph(CustomArray::CSR_Matrix<Weight>& edges, Vertex vertices,
					bool directed)
			: AdjMat(edges)
		{
			if(vertices < 2)
				throw vertices; // Trivial graph of only one element
			else if(edges.GetRowCount() != edges.GetColumnCount() ||
					edges.GetColumnCount() != vertices)
				throw edges; // Invalid format of matrix
			VertexCount = vertices;
			IsDirected	= directed;
		}

		/**
		 * @brief This function checks
		 * if the graph is a directed
		 * graph or not
		 *
		 * @return bool: True if
		 * Graph is directed, otherwise
		 * undirected
		 */
		bool isDirected() { return IsDirected; }

		/**
		 * @brief Get the number of Vertices `V`
		 * in this graph. This graph will contain
		 * vertices 0, 1, 2, ... V-1
		 *
		 * @return int Number of vertices
		 */
		Vertex get_Vertices() { return VertexCount; }

		/**
		 * @brief Get the Edge Count for
		 * the graph
		 *
		 * @return Vertex: The number of edges of the graph
		 */
		Vertex get_Edge_Count() { return AdjMat.Non_Zero_Elements_Count(); }

		/**
		 * @brief Get the Weight between
		 * two vertices u and v
		 *
		 * @param u A vertex in this graph
		 * @param v A vertex in this graph
		 * @return uint The weight of the edge
		 */
		Weight get_Weight(Vertex u, Vertex v)
		{
			if(u == v)
				return 0;
			if(IsDirected == false && u > v)
				return AdjMat.Get(v, u);
			return AdjMat.Get(u, v);
		}

		/**
		 * @brief |NOT SUPPORTED| Add an vertex in the graph
		 *
		 * @param v: The vertex to add to the graph
		 * @return bool : true means that vertex addition was successful,
		 * otherwise vertex addition failed/not supported
		 */
		bool add_Vertex(Vertex v) { return false; }

		/**
		 * @brief |NOT SUPPORTED| Removes a vertex (and any associated edges)
		 * from the graph
		 *
		 * @param v the vertex to remove
		 * @return bool: true means that the vertex removal was successful,
		 * otherwise vertex removal failed/not supported
		 */
		bool remove_Vertex(Vertex v) { return false; }

		/**
		 * @brief |NOT SUPPORTED| Updates an edge of the graph
		 *
		 * @param u the vertex from where the edge begins
		 * @param v the vertex on which the edge ends
		 * @param w the weight of the edge. If 0, the edge is set for removal,
		 * otherwise the edge is set to be added to the graph
		 *
		 * @return bool: true means that the edge updation was successful,
		 * otherwise edge updation failed/not supported
		 */
		bool update_Edge(Vertex u, Vertex v, Weight w) { return false; }

		/**
		 * @brief Iterates all the edges with their neighbouring vertices
		 * for a given vertex in the graph
		 *
		 * @param v The vertex whose neighbours are to be iterated
		 * @param fn The function to execute on each neighbour
		 * @param mode The mode of iteration. 1 for only neighbours where edge
		 * is incoming to v  (that is, all edges (x,v), where v is the queried
		 * vertex), 2 for only neighbours to which edge is outgoing from v (that
		 * is, all edges (v,x), where v is * the queried vertex), and 3 for
		 * both. In mode 3, the nodes that are both incoming and outgoing
		 * neighbours get iterated twice.
		 */
		void iterate_Neighbours(
			Vertex v,
			std::function<void(const Vertex u, const Vertex v, const Weight w)>
				fn,
			int mode = 3)
		{
			int i, j, ele = AdjMat.Non_Zero_Elements_Count();
			if(ele == 0)
				return;
			if(mode < 1 || mode > 3)
				mode = 3;
			Vertex st = AdjMat.row_index[v], ed = AdjMat.row_index[v + 1];
			if(mode & 1) // outgoing neighbours
			{
				j				   = AdjMat.row_index[1];
				Vertex current_row = 0;
				for(i = 0; i < ele; i++)
				{
					while(j <= i)
					{
						current_row++;
						j = AdjMat.row_index[current_row + 1];
					}
					if(AdjMat.col_index[i] == v)
					{
						fn(current_row, v, AdjMat.NNZ[i]);
					}
				}
			}
			if(mode & 2) // incoming neighbours
			{
				for(i = st, j = ed; i < j; i++)
				{
					fn(v, AdjMat.col_index[i], AdjMat.NNZ[i]);
				}
			}
		}

		/**
		 * @brief Iterates all the edges within the graph
		 *
		 * @param fn The function to execute on each edge
		 */
		void iterate_Edges(
			std::function<void(const Vertex u, const Vertex v, const Weight w)>
				fn)
		{
			AdjMat.Iterate([&fn](const Weight item, const Vertex x,
								 const Vertex y) -> void { fn(x, y, item); });
		}
	};

	/**
	 * @brief Generates a random graph with density greater than 0.5
	 *
	 * @param vertices
	 * @param vertices The number of vertices supposed to be in the graph
	 * @param edges The number of edges supposed to be in the graph
	 * @param directed Controls the directedness of the graph
	 * @return Returns a pointer to the generated graph
	 */
	CustomArray::Array1D<GraphEdge>*
	GenerateRandomDenseGraph(Vertex vertices, Vertex edges, bool directed,
							 std::ostream& logger)
	{
		random_wrapper				 generator(0, vertices - 1);
		CustomArray::Array2D<Weight> Adjmat(vertices, vertices);
		Vertex						 i, j;
		Weight						 w;
		for(i = 0; i < vertices; i++)
		{
			for(j = vertices - 1; j > 0; j--)
			{
				if(i == j)
				{
					if(directed)
						continue;
					else
						break;
				}
				w = generator.generate() + 1;
				if(directed)
					Adjmat.Set(i, j, w);
				else
				{
					Adjmat.Set(i, j, w);
					Adjmat.Set(j, i, w);
				}
			}
		}
		CustomArray::Array1D<GraphEdge>* edge_list =
			new CustomArray::Array1D<GraphEdge>(edges);
		Vertex total = vertices * (vertices - 1);
		if(!directed)
			total >>= 1;
		total -= edges;
		while(total > 0)
		{
			i = generator.generate();
			j = generator.generate();
			if(Adjmat.Get(i, j) != 0)
			{
				if(directed)
					Adjmat.Set(i, j, 0);
				else
				{
					Adjmat.Set(i, j, 0);
					Adjmat.Set(j, i, 0);
				}
				total--;
			}
		}
		for(i = total = 0; i < vertices; i++)
		{
			for(j = vertices - 1; j > 0; j--)
			{
				if(i == j)
				{
					if(directed)
						continue;
					else
						break;
				}
				try
				{
					w = Adjmat.Get(i, j);
					if(w != 0)
					{
						edge_list->Set(total++, GraphEdge(i, j, w));
						#if DEBUG
						std::cout << "Writing edge (" << i << ", " << j
								  << ")...with weight=" << w << "\t\r";
						#endif
					}
				}
				catch(std::exception& e)
				{
					std::cerr << e.what() << "\n*\n*\n";
					std::cout << e.what() << "\n*\n*\n";
					throw e;
				}
			}
		}
		return edge_list;
	}

	/**
	 * @brief Generates a random sparse graph with density less than 0.5
	 *
	 * @param vertices The number of vertices in the graph
	 * @param edges The number of edges that should be in the graph
	 * @param directed The directedness of the graph
	 * @param logger The std::ostream logger to log the details of the generated
	 * graph
	 * @return CustomArray::Array1D<GraphEdge>* The pointer to the generated
	 * graph
	 */
	CustomArray::Array1D<GraphEdge>*
	GenerateRandomSparseGraph(Vertex vertices, Vertex edges, bool directed,
							  std::ostream& logger)
	{
		struct KeyGen
		{
			Weight Merge(Vertex a, Vertex b)
			{
				if(a > b)
				{
					Vertex c = a;
					a		 = b;
					b		 = c;
				}
				Weight x = a;
				x		 = (x << 32) + b;
				return x;
			}
		};

		KeyGen							 keygen;
		Vertex*							 all_nodes = MALLOC<Vertex>(vertices);
		std::unordered_set<Weight>		 all_edges;
		Vertex							 i, j, k;
		CustomArray::Array1D<GraphEdge>* edgelist =
			new CustomArray::Array1D<GraphEdge>(edges);
		logger << "Preparing tree with nodes = " << vertices << "\n";
		vertices--;
		random_wrapper node_picker(0, vertices);
		for(i = 0; i <= vertices; i++)
			all_nodes[i] = i;
		std::random_shuffle(all_nodes, all_nodes + i);
		// 1. Prepare a tree with V-1 random edges such that the graph is
		// connected
		for(i = 1; i <= vertices; i++)
		{
			j = node_picker.generate() % i;
			j = all_nodes[j]; // Select a random node from the MST set
			k = all_nodes[i]; // Random node from set that isn't connected (yet)
			if(directed || j < k) // Add the edge to the list
			{
				edgelist->Set(i - 1,
							  GraphEdge(j, k, node_picker.generate() + 1));
				all_edges.insert(keygen.Merge(j, k));
			}
			else
			{
				edgelist->Set(i - 1,
							  GraphEdge(k, j, node_picker.generate() + 1));
				all_edges.insert(keygen.Merge(k, j));
			}
		}
		logger << "Total edges required: " << edges << '\n';
		edges = edges - vertices;
		std::random_shuffle(all_nodes, all_nodes + i);
		// 2. Add random additional edges
		int x = vertices;
		while(edges > 0)
		{
			j = node_picker.generate();
			k = node_picker.generate();
			if(j == k)
			{
				k++;
				if(k > vertices)
					k = 0;
			}
			if(directed == false && j > k)
			{
				i = j;
				j = k;
				k = i;
			}
			if(all_edges.count(keygen.Merge(j, k)) == 0)
			{
				edgelist->Set(x++, GraphEdge(j, k, 1 + node_picker.generate()));
				all_edges.insert(keygen.Merge(j, k));
				edges--;
			}
			#if DEBUG
			logger << "\tTotal edges remaining: " << edges << "\t\r";
			#endif
		}
		free(all_nodes);
		logger << "\nThe graph is generated successfully!\n";
		return edgelist;
	}

	/**
	 * @brief Generates a random graph with the given details
	 *
	 * @param vertices The number of vertices supposed to be in the graph
	 * @param edges The number of edges supposed to be in the graph
	 * @param directed The directedness of the graph
	 * @param logger The std::ostream object to log the details of the function
	 * @return CustomArray::Array1D<GraphEdge>* The pointer to the created graph
	 */
	CustomArray::Array1D<GraphEdge>* GenerateRandomGraph(Vertex		   vertices,
														 Vertex		   edges,
														 bool		   directed,
														 std::ostream& logger)
	{
		Vertex max_edges = vertices * (vertices - 1);
		if(directed == false)
			max_edges = max_edges >> 1;
		if(edges > max_edges)
			edges = max_edges;
		else if(edges < vertices - 1)
			edges = vertices - 1;
		if((edges << 1) < max_edges) // if density is less than/equal to 0.5
			return GenerateRandomSparseGraph(vertices, edges, directed, logger);
		else
			return GenerateRandomDenseGraph(vertices, edges, directed, logger);
	}

	/**
	 * @brief Generates a random graph with randomly picked number of vertices
	 *
	 * @param logger The std::ostream object to log the details within
	 * @param vertices The number of vertices in the graph
	 * @param directed The directedness of the graph
	 * @return CustomArray::Array1D<GraphEdge>* The pointer to the created graph
	 */
	CustomArray::Array1D<GraphEdge>* GenerateRandomGraph(std::ostream& logger,
														 Vertex		   vertices,
														 bool directed = false)
	{
		int max_edges = vertices * (vertices - 1);
		if(directed == false)
			max_edges = max_edges >> 1;
		random_wrapper edges(vertices, max_edges);
		return GenerateRandomGraph(vertices, edges.generate(), directed,
								   logger);
	}

	/**
	 * @brief Generates a random graph that is a tree
	 *
	 * @param logger The logger to log the details
	 * @param vertices The number of vertices in the graph
	 * @param directed The directedness of the graph
	 * @return CustomArray::Array1D<GraphEdge>* The pointer to the created graph
	 */
	CustomArray::Array1D<GraphEdge>* GenerateRandomTree(std::ostream& logger,
														Vertex		  vertices,
														bool directed = false)
	{
		return GenerateRandomGraph(vertices, vertices - 1, directed, logger);
	}

	/**
	 * @brief Writes the graph to a given std::ostream file
	 *
	 * @param edge_list The list of edges (as a graph) to write
	 * @param vertices The number of vertices in the graph
	 * @param o The std::ostream file to write to
	 */
	void SaveGeneratedGraph(CustomArray::Array1D<GraphEdge>* edge_list,
							Vertex vertices, std::ostream& o)
	{
		Weight edges = edge_list->GetCount();
		o << vertices << " " << edges << "\n";
		edge_list->Iterate([&o](const GraphEdge g) -> void
						   { o << (g.u) << " " << (g.v) << " " << g.w << "\n"; });
	}
	void SaveGeneratedGraphRt(CustomArray::Array1D<GraphEdge>* edge_list,
							Vertex vertices, std::ostream& o)
	{
		Weight edges = edge_list->GetCount();
		o <<"p sp "<< vertices << " " << edges << "\n";
		edge_list->Iterate([&o](const GraphEdge g) -> void
						   { o << "a " << (g.u + 1) << " " << (g.v + 1) << " " << g.w << "\n"; });
	}

	/**
	 * @brief Saves/Prints the given graph on the std::ostream object
	 *
	 * @param g The graph to save/print
	 * @param o The std::ostream object to save the graph at
	 */
	void SaveGeneratedGraph(graphADT* g, std::ostream& o)
	{
		Vertex vertices = g->get_Vertices();
		Weight count	= 0;
		g->iterate_Edges([&count](const Vertex u, const Vertex v,
								  const Weight w) -> void { count++; });
		o << vertices << " " << count << "\n";
		g->iterate_Edges(
			[&o](const Vertex u, const Vertex v, const Weight w) -> void
			{ o << u << " " << v << " " << w << "\n"; });
	}

	/**
	 * @brief Reads the edgelist from a file
	 *
	 * @param inp The input file
	 * @param vertices The vertices in the graph
	 * @param edges The edges in the graph
	 * @return CustomArray::Array1D<GraphEdge>& The reference to the
	 * array of edges
	 */
	CustomArray::Array1D<GraphEdge>&
	ReadEdgeList(std::istream& inp, Vertex& vertices, Weight& edges)
	{
		Vertex u, v, t;
		Weight w, p = 0;
		inp >> vertices >> edges;
		CustomArray::Array1D<GraphEdge>* edge_list =
			new CustomArray::Array1D<GraphEdge>(edges);
		while(edges-- > 0)
		{
			inp >> u >> v >> w;
			if(u > v)
			{
				t = u;
				u = v;
				v = t;
			}
			edge_list->Set(p++, GraphEdge(u, v, w));
		}
		edges = edge_list->GetCount();
		return *edge_list;
	}

	/**
	 * @brief Represents a graph that is
	 * obtained (partially) via a stream
	 * 
	 */
	class AbstractGraphStreamReader
	{
		public:
		/**
		 * @brief Checks if there are more edges
		 * that can be loaded
		 * 
		 * @return true More edges are remaining to load
		 * @return false No more edges remain that can be loaded
		 */
		virtual bool EdgesRemainaing() = 0;

		/**
		 * @brief Reads the next edge from the stream
		 * 
		 * @return GraphEdge 
		 */
		virtual GraphEdge ReadNextEdge() = 0;
	};

	enum FileTypes:uint8_t 
	{
		FileTypes_Default, 
		FileTypes_MatrixMarket,
		FileTypes_Gr
	};

	class FileGraphReader : public AbstractGraphStreamReader
	{
	 private:
		uint8_t file_type;
		bool weighted;
		std::ifstream file;
		Vertex vertices;
		Weight edges, remaining;
		random_wrapper rng;
	 public:
	 	FileGraphReader(std::string path, uint8_t type=0, 
			bool wtd=true) : file(path), rng(1, 100)
		{
			file_type = type;
			weighted = wtd;
			char c, w = file_type == FileTypes_MatrixMarket ? '%' : 'c';
			std::string dummy;
			switch (file_type) 
			{
				case FileTypes_Default: // Default format
					file >> vertices >> edges;
					std::cout << "edges " << edges << std::endl;
				break;
				case FileTypes_MatrixMarket:
				case FileTypes_Gr:
				default:
					while(file >> c)
					{
						if(c != w)
						{
							file >> dummy >> vertices >> edges;
							break;
						}
						else
							getline(file, dummy);
					}
				break;
			}
			remaining = edges;
		}
		/**
		 * @brief Get the number of Vertices 
		 * 
		 * @return Vertex number of vertices
		 */
		Vertex GetVertexCount()
		{
			return vertices;
		}
		/**
		 * @brief Get the number of Edges 
		 * 
		 * @return Weight number of edges
		 */
		Weight GetEdgesCount()
		{
			return edges;
		}
		/**
		 * @brief Checks if there are more edges
		 * that can be loaded
		 * 
		 * @return true More edges are remaining to load
		 * @return false No more edges remain that can be loaded
		 */
		bool EdgesRemainaing()
		{
			return remaining > 0;
		}

		/**
		 * @brief Reads the next edge from the stream
		 * 
		 * @return GraphEdge 
		 */
		GraphEdge ReadNextEdge()
		{
			GraphEdge edge;
			switch (file_type) 
			{
				default:
				case FileTypes_Default: // Default format
					file >> edge.u >> edge.v; 
					if(weighted) 
						file >> edge.w;
					else
						edge.w = rng.generate();
				break;
				case FileTypes_MatrixMarket: // Matrix Market format
					double d;
					file >> edge.u >> edge.v; 
					if(weighted) 
						file >> d;
					else
						d = rng.generate();
					edge.w = (Weight)d;
					edge.u--;
					edge.v--;
				break;
				case FileTypes_Gr:
					char c;
					std::string dummy;
					while(file >> c)
					{
						if(c == 'a')
						{
							file >> edge.u >> edge.v;
							if(weighted)
								file >> edge.w;
							else
								edge.w = rng.generate();
							edge.u--;
							edge.v--;
							break;
						}
						else
							getline(file, dummy);
					}
				break;
			}
			remaining--;
			return edge;
		}
	};

	CustomArray::Array1D<GraphEdge>& GetEdgesFromFile(std::string path,
		uint8_t file_type, Vertex &vertices, bool wtd=true, int percent=0)
	{
		FileGraphReader stream(path, file_type, wtd);
		vertices = stream.GetVertexCount();
		CustomArray::Array1D<GraphEdge> *array;
		Weight numEdgesTotal = stream.GetEdgesCount();
    	float percentEdge = (float(percent) / 100) * numEdgesTotal;
		numEdgesTotal = Weight(percentEdge);
		//numEdgesTotal *= 2;
		std::cout << "Percent gpu " << numEdgesTotal << std::endl;
		//std::cout << "Percent " << percent<< std::endl;
		array = new CustomArray::Array1D<GraphEdge>(numEdgesTotal);
		Weight i=0;
		Weight done=0;
		while(stream.EdgesRemainaing() && i<numEdgesTotal)
		{
			//array->Set(done++, stream.ReadNextEdge());
			GraphEdge g = stream.ReadNextEdge();
			GraphEdge g_inv;
			//g_inv.u = g.v;
			//g_inv.v = g.u;
			//g_inv.w = g.w;
			array->Set(done++, g);
			//array->Set(done++, g_inv);
			i++;
		}
		std::cout << "done " << std::endl;
		return *array;
	}


	CustomArray::Array1D<GraphEdge>& GetEdgesFromFileNew(std::string path,
		uint8_t file_type, Vertex &vertices, bool wtd=true, int percent=0, int offset=0)
	{
		FileGraphReader stream(path, file_type, wtd);
		vertices = stream.GetVertexCount();
		CustomArray::Array1D<GraphEdge> *array;
		Weight numEdgesTotal = stream.GetEdgesCount();
    	float percentEdge = (float(percent) / 100) * numEdgesTotal;
		numEdgesTotal = Weight(percentEdge);
		//numEdgesTotal *= 2;
		std::cout << "Percent gpu " << numEdgesTotal << std::endl;
		//std::cout << "Percent " << percent<< std::endl;
		array = new CustomArray::Array1D<GraphEdge>(numEdgesTotal);
		Weight i=0;
		Weight done=0;
		if(offset!=0){
			while(stream.EdgesRemainaing() && i<offset)
			{
				GraphEdge g = stream.ReadNextEdge();
				i++;
			}
		}
		while(stream.EdgesRemainaing() && i<numEdgesTotal)
		{
			//array->Set(done++, stream.ReadNextEdge());
			GraphEdge g = stream.ReadNextEdge();
			GraphEdge g_inv;
			//g_inv.u = g.v;
			//g_inv.v = g.u;
			//g_inv.w = g.w;
			array->Set(done++, g);
			//array->Set(done++, g_inv);
			i++;
		}
		std::cout << "done " << std::endl;
		return *array;
	}



	SparseGraph* GetGraphFromFileNew(std::string path,
		uint8_t file_type, bool wtd=true, int percent=0, int offset=0)
	{
		Vertex v;
		auto array = GetEdgesFromFile(path, file_type, v, wtd, percent);
		std::cout << "done edges from file " << std::endl;
		SparseGraph* sp_graph = new SparseGraph(array, v, false);	
		return sp_graph;
	}


	SparseGraph* GetGraphFromFile(std::string path,
		uint8_t file_type, bool wtd=true, int percent=0)
	{
		Vertex v;
		auto array = GetEdgesFromFile(path, file_type, v, wtd, percent);
		std::cout << "done edges from file " << std::endl;
		SparseGraph* sp_graph = new SparseGraph(array, v, false);	
		return sp_graph;
	}

#pragma endregion SparseGraphComponents

	/// @brief  Graph algorithms
	namespace Algo
	{
		/// @brief Encapsulates the result of MST obtained from a graph
		struct MST_Result
		{
			/// @brief The number of edges in the EdgeList
			Vertex Edge_Count;

			/// @brief The number of (disconnected) components in the graph
			Vertex Components;

			/// @brief The total cost of the MST
			Weight Total_Cost;

			/// @brief The list of selected edges.
			CustomArray::Array1D<GraphEdge>* EdgeList;

			/// @brief Returns the number of vertices in the original graph
			/// @return The numbe of vertices in the original graph
			Vertex Graph_Vertices() { return Edge_Count + Components; }
		};

		#pragma region MST_Serial

		/**
		 * @brief Prim algorithm implemented using a Fibonacci Heap
		 *
		 * @param graph The graph to compute MST of
		 * @param cost The output parameter (passed by reference) which
		 * is the cost of the MST. If the graph is not connected, cost is
		 * programmed to be 0
		 * @return CustomArray::Array1D<GraphEdge>* The list of edges
		 * selected in the MST
		 */
		MST_Result MST_Serial(graphADT* graph)
		{
			Vertex	   vertices = graph->get_Vertices();
			MST_Result result;
			result.Edge_Count = 0;
			result.Components = 1;
			result.EdgeList =
				new CustomArray::Array1D<GraphEdge>(vertices - 1);
			result.Total_Cost = 0;
			if(vertices < 2 || graph->isDirected())
				return result;
			Weight inf = 0;
			inf--;
			Vertex *Pred = MALLOC<Vertex>(vertices), key, done, pred;
			auto	keys = MALLOC<Heap::FibonacciNode<Vertex, Weight>*>(vertices);
			Heap::FibonacciHeap<Vertex, Weight> Min_Dist_Heap;
			#if DEBUG
			std::cout << "Beginning Prim algorithm...\n";
			clock_t s1, s2;
			double	time, total_time = 0, avg_tpn, etc;
			#endif
			for(Vertex i = 1; i < vertices; i++)
			{
				Pred[i] = vertices + 1;
				keys[i] = Min_Dist_Heap.Insert(i, inf);
			}
			// Add Vertex 0 to the MST
			keys[0] = Min_Dist_Heap.Insert(0, 0);
			Pred[0] = vertices;
			graph->iterate_Neighbours(
				0,
				[&](const Vertex u, const Vertex v, const Weight w) -> void
				{
					Vertex x = u == 0 ? v : u;
					Min_Dist_Heap.DecreaseKey(keys[x], w);
					Pred[x] = 0;
				},
				3);
			done = 0;
			#if DEBUG
			std::cout << "Strting from node 0...\n";
			s1 = clock();
			#endif
			Min_Dist_Heap.DeleteMin(); // Remove 0
			Weight dist;
			do
			{
				key	 = Min_Dist_Heap.GetTopKey();
				dist = Min_Dist_Heap.GetTop();
				pred = Pred[key];
				Min_Dist_Heap.DeleteMin();
				keys[key] = nullptr;
				Pred[key] = vertices;
				if(pred >= vertices) // unreachable
				{
					result.Components++;
					graph->iterate_Neighbours(
						key,
						[&](const Vertex u, const Vertex v,
							const Weight w) -> void
						{
							Vertex x = u == key ? v : u;
							Min_Dist_Heap.DecreaseKey(keys[x], w);
							Pred[x] = key;
						},
						3);
					continue;
				}
				if(key > pred)
					result.EdgeList->Set(done, GraphEdge(pred, key, dist));
				else
					result.EdgeList->Set(done, GraphEdge(key, pred, dist));
				done++;
				#if DEBUG
				s2	 = clock();
				time = ((double)(s2 - s1)) / CLOCKS_PER_SEC;
				total_time += time;
				avg_tpn = total_time / done;
				etc		= (int)((vertices - 1 - done) * avg_tpn);
				std::cout << "Processed " << done << '/' << vertices
						  << " nodes; Avg speed = " << avg_tpn
						  << " per node, ETC = " << etc << " seconds \r";
				s1 = clock();
				#endif
				result.Total_Cost += dist;
				graph->iterate_Neighbours(
					key,
					[&](const Vertex u, const Vertex v, const Weight w) -> void
					{
						Vertex x = u == key ? v : u;
						if(Pred[x] == vertices)
							return;
						else
						{
							if(Min_Dist_Heap.DecreaseKey(keys[x], w))
								Pred[x] = key;
						}
					},
					3);
			} while(Min_Dist_Heap.GetSize() > 0);
			#if DEBUG
			std::cout << "Memory cleanup...\n";
			#endif
			free(keys);
			free(Pred);
			return result;
		}

		#pragma endregion MST_Serial

		namespace CMST_Kernels
		{
			const int	 VertexBits = 8 * sizeof(Vertex);
			const int	 WeightBits = 8 * sizeof(Weight);
			const Weight lsb_mask	= 0x00000000ffffffff;
			const Weight msb_mask	= 0xffffffff00000000;
			const Weight invalidW	= 0xffffffffffffffff;
			typedef thrust::tuple<Weight, Weight> W_tuple;

			/**
			 * @brief Binary operation required for reduction
			 * with thrust::reduce_by_key()
			 *
			 */
			struct Min_op
			{
				__host__ __device__ W_tuple operator()(const W_tuple& a,
													   const W_tuple& b) const
				{
					Weight aw = thrust::get<0>(a), bw = thrust::get<0>(b);
					return aw < bw ? a : b;
				}
			};

			/**
			 * @brief Merges the value of two vertices
			 * into a single uint64_t object
			 *
			 * @param p The lower value
			 * @param q The greater vaulue
			 * @return Weight the merged value
			 */
			__host__ __device__ Weight Merge(Vertex a, Vertex b)
			{
				Weight x = a;
				x		 = (x << VertexBits) + b;
				return x;
			}

			/**
			 * @brief Unbinds two merged vertices from
			 * a single uint64_t object
			 *
			 * @param x The merged object
			 * @param a The lower value, passed by reference
			 * @param b The greater value, passed by reference
			 */
			__host__ __device__ void Unmerge(Weight w, Vertex& a, Vertex& b)
			{
				Weight x = w;
				Weight y = x & lsb_mask;
				x		 = x >> VertexBits;
				b		 = (Vertex)y;
				a		 = (Vertex)x;
			}

			/**
			 * @brief Structure for Uncompressed Edge
			 *
			 */
			struct UncompressedEdge
			{
				/**
				 * @brief Endpoints of the edge, size |E|
				 *
				 */
				Weight* uv;
				/**
				 * @brief Weight of the edge, size |W|
				 *
				 */
				Weight* w;
			};

			/**
			 * @brief Pair of nodes and weight corresponding
			 * to some edge
			 *
			 */
			struct AdjListNode
			{
				/**
				 * @brief Destination/Outgoing node
				 *
				 */
				Vertex* node;
				/**
				 * @brief The source vertex
				 *
				 */
				Vertex* origin;
				/**
				 * @brief The weight of the edge
				 *
				 */
				Weight* weight;
			};

			/**
			 * @brief A C-style struct containing
			 * a CSR matrix
			 *
			 */
			struct CSR_Wrapper
			{
				/**
				 * @brief Non-zero elements of the matrix
				 *
				 */
				Weight* NNZ;
				/**
				 * @brief Column indices
				 *
				 */
				Vertex* C_idx;
				/**
				 * @brief Row cumelative counts
				 *
				 */
				Vertex* R_cct;
			};

			/**
			 * @brief Initializes the Edge List for the graph.
			 * One thread per ordered edge is launched
			 *
			 * @param vertices The number of vertices
			 * @param old The old CSR matrix
			 * @param edges The number of undirected edges
			 * @param UCM The uncompressed edge list
			 * @param cur The current CSR matrix
			 */
			__global__ void getInitEdgeList(const Vertex	  vertices,
											const Vertex	  edges,
											const CSR_Wrapper old,
											UncompressedEdge  UCM,
											CSR_Wrapper		  cur)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// calculate thread id
				if(tid < vertices) // check if threads do not exceed the number
								   // of vertices
				{
					Vertex st = old.R_cct[tid], ed = old.R_cct[tid + 1], pid;
					Weight y;
					while(st < ed)
					{
						y					= old.NNZ[st];
						pid					= old.C_idx[st];
						UCM.uv[st + edges]	= Merge(tid, pid);
						cur.NNZ[st + edges] = y;
						UCM.uv[st]			= Merge(pid, tid);
						cur.NNZ[st++]		= y;
					}
				} // Created list of encoded edges
			}

			/**
			 * @brief Prepares the modified CSR_Matrix for the CUDA MST
			 * function. One thread per unordered edge is launched.
			 *
			 * @param vertices The number of vertices
			 * @param new_edges The number of directed edges
			 * @param UCM The uncompressed Edge List
			 * @param cur The current CSR_matrix
			 */
			__global__ void getInitCsrMatrix(const Vertex			vertices,
											 const Vertex			new_edges,
											 const UncompressedEdge UCM,
											 CSR_Wrapper			cur)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < new_edges) // Thread is loaded per edge
				{
					Vertex p, q; // The vertices obtained from unmerging

					Weight w = UCM.uv[tid];
					// Obtain the merge pair

					Unmerge(w, p, q);
					// Obtain the merged vertices at this position

					cur.C_idx[tid] = q; // C_idx is set for every edge
					if(tid == 0)		// Not going to update R_cct[tid]...
					{
						cur.R_cct[0] = 0;
						// ...so special cases for CSR matrix are handled

						cur.R_cct[vertices] = new_edges;
						// works for tid = 0, and tid = new_edges
					}
					else
					{
						Vertex r; // The vertex at the previous position

						Unmerge(UCM.uv[tid - 1], r, q);
						// r is obtained from unmerge. q is not needed

						if(r == p - 1) // If graph is connected, r=p-1, or r=p
							cur.R_cct[p] = tid;
						// When r = p-1, we update this position for  R_cct

						else if(p != r) // At least one 0-degree vertex exists
						{
							while(r < p) // Set values for all 0-degree vertices
								cur.R_cct[++r] = tid;
						}
					}
				}
			}

			/**
			 * @brief Kernel for Boruvka approach. Finds the successor, i.e
			 * closest vertex to the current vertex. One thread per vertex is
			 * launched.
			 *
			 * @param vertices The number of vertices
			 * @param csr The edges wrapped in CSR matrix
			 * @param nearest The successor/nearest neighbour for each vertex
			 * @param UCM the uncompressed edge pairs
			 */
			__global__ void FindSuccessor(const Vertex		vertices,
										  const CSR_Wrapper csr,
										  AdjListNode		nearest,
										  UncompressedEdge	UCM)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < vertices) // if the vertex with this thread id exists
				{
					Vertex st = csr.R_cct[tid], ed = csr.R_cct[tid + 1], i;
					// The indices for neighbour of this vertex is [st, ed)

					Vertex closest	= tid;		// default/invalid result
					Weight min_dist = invalidW; // default/invalid result
					// finally set the value for the nearest node

					UCM.w[tid] = invalidW;	 // invalid value for UV pair
					for(i = st; i < ed; i++) // iterate for all neighbours
					{
						if(min_dist > csr.NNZ[i]) // if minimum edge is found
						{
							min_dist   = csr.NNZ[i];   // update minimum dist
							closest	   = csr.C_idx[i]; // update minimum vertex
							UCM.w[tid] = UCM.uv[i];
							// index of this selected edge
						}
					}
					// For 0-degree node, UCM.w = inf, closest = tid
					nearest.node[tid] = closest;
					// finally set the value for the nearest node
					nearest.weight[tid] = min_dist;
					// finally set the value of weight of this edge
				}
			}

			/**
			 * @brief Kernel for Boruvka approach. Removes cycles. It is
			 * guaranteed that all cycles contain exactly 2 vertices. This is
			 * easy to detect, and a self loop is formed. Thus a forest of trees
			 * are formed by this kernel. One thread per vertex is launched.
			 *
			 * @param vertices The number of vertices
			 * @param nearest The successor/nearest neighbour for each vertex
			 */
			__global__ void RemoveCycles(const Vertex vertices,
										 AdjListNode  nearest,
										 Weight*	  selWeights)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < vertices)
				// Only run if for the vertex with this thread id exists
				{
					// Copy the edge_id and the weight of this edge
					int parent = nearest.node[tid]; // parent of node is found
					if(parent == tid)				// 0-degree node
					{
						nearest.node[tid] = vertices;
						// This will push this item at the end while sorting
						selWeights[tid] = nearest.weight[tid];
					}
					else
					{
						int grandparent = nearest.node[parent];
						// parent of parent

						parent =
							(grandparent == tid && tid < parent) ? tid : parent;
						// Since cycle must contain exactly 2 vertex, a cycle
						// is found as such. Also works for 0-degree node

						nearest.node[tid] = parent;
						// One vertex is selected to be the root of the tree.
						// For the root, it is its own parent. Other vertices
						// are unchanged.

						selWeights[tid] =
							tid == parent ? 0 : nearest.weight[tid];
						// This node is repeated twice, so it is made
						// to be 0, and the cost of tree will be
						// evaluated correctly
					}
				}
			}

			/**
			 * @brief Kernel for Boruvka apprach. At the end of this kernel, all
			 * vertices in a given tree point to the root of the tree instead of
			 * its direct parent. One thread per vertex is launched
			 *
			 * @param vertices The number of vertices
			 * @param nearest The array containing the parent of the ith node
			 */
			__global__ void UpdateRoot(const Vertex	 vertices,
									   const Vertex* nearest_node,
									   Vertex*		 cluster)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < vertices)
				// Only run if for the vertex with this thread id exists
				{
					int root = nearest_node[tid];
					int iter = 0;
					// Find the root of this vertex
					if(root < vertices) // Not a 0-degree node
					{
						// Root is its own parent.
						while(root != nearest_node[root] & iter < 500){
							root = nearest_node[root];
							iter++;
						}
							
						// Search until the root is found
					}
					cluster[tid] = root;
					// Assign the root of the tree to this vertex
				}
			}

			/**
			 * @brief Kernel for Boruvka apprach. At the end of this kernel, all
			 * super-vertices initializes a lookup for their values of vertices
			 * in the previous iteration. One thread per vertex is launched.
			 *
			 * @param RootCount The number of super vertices.
			 * @param Roots Value of the previous vertex for this supervertex
			 * @param Mapper The lookup table to map vertex to supervertex
			 */
			__global__ void CreateIndexMapper(const Vertex	RootCount,
											  const Vertex* Roots,
											  Vertex*		Mapper)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < RootCount)
				// Only run if for the vertex with this thread id exists
				{
					Mapper[Roots[tid]] = tid;
				}
			}

			/**
			 * @brief Kernel for Boruvka apprach. At the end of this kernel, all
			 * vertices are assigned their supervertex values of vertices given
			 * by the mapper lookup table. One thread per vertex is launched.
			 *
			 * @param RootCount The number of super vertices.
			 * @param Roots Value of the previous vertex for this supervertex
			 * @param Mapper The lookup table to map vertex to supervertex
			 */
			__global__ void MapRootToIndex(const Vertex	 vertices,
										   const Vertex* mapper,
										   Vertex*		 rootofVertex)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < vertices && rootofVertex[tid] < vertices)
				{
					rootofVertex[tid] = mapper[rootofVertex[tid]];
				}
			}

			/**
			 * @brief Kernel for Boruvka apprach. Create new uncompressed
			 * adjoint matrix from the given old list and cluster. One thread
			 * per old vertex is launched.
			 *
			 * @param old_vertices The number of vertices in the old graph
			 * @param old The edgelist of the old graph
			 * @param roots The clusters id of each vertex
			 * @param UCM The Uncompressed format for vertices
			 */
			__global__ void NewEdgeList_k1(const Vertex		old_vertices,
										   const Vertex		new_vertices,
										   CSR_Wrapper		old,
										   const Vertex*	cluster,
										   UncompressedEdge UCM)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < old_vertices)
				// Only run if for the vertex with this thread id exists
				{
					Vertex st = old.R_cct[tid], ed = old.R_cct[tid + 1],
						   mycluster = cluster[tid];
					// mycluster is the super-vertex of  tid vertex
					for(Vertex i = st, k, j; i < ed; i++)
					// Iterate for every edge of vertex `tid`
					{
						j = old.C_idx[i]; // This edge is (tid, j)
						k = cluster[j];	  // k is the supervertex of vertex j
						UCM.w[i] = old.NNZ[i]; // Weight is unchanged
						// edge between (mycluster, j)
						if(k != mycluster && k < old_vertices &&
						   mycluster < old_vertices)
							// Only considering edges between  different
							// clusters, and adding to list of uncompressed edge
							UCM.uv[i] = CMST_Kernels::Merge(mycluster, k);
						// Vertices are merged in uv
						else
							UCM.uv[i] = invalidW;
						// now it is sent to the last index on sort
					}
				}
			}

			/**
			 * @brief Kernel for Boruvka apprach. Prepares the R_cct and C_idx
			 * elements of the new CSR matrix. Number of threads must be equal
			 * to (new_edges + 1) for this kernel.
			 *
			 * @param new_edges The number of new edges
			 * @param UCM The Uncompressed Edge List
			 * @param cur_vertices The number of super-vertices
			 * @param cur The current CSR_Matrix to set
			 */
			__global__ void
			NewEdgeList_k2(const Vertex new_edges, const UncompressedEdge UCopy,
						   const Weight* uv, const Vertex cur_vertices,
						   CSR_Wrapper cur, UncompressedEdge UCM)
			{
				Vertex tid = (blockIdx.x * blockDim.x) + threadIdx.x;
				// find thread id
				if(tid < new_edges) // Thread is loaded per edge
				{
					Vertex p, q; // The vertices obtained from unmerging
					Unmerge(uv[tid], p, q);
					// Obtain the merged vertices at this position
					UCM.uv[tid] = cur.NNZ[tid];
					// Copy the corresponding edge id back to UCM
					cur.NNZ[tid] = UCopy.w[tid]; // Value is set for every edge
					cur.C_idx[tid] = q;			 // C_idx is set for every edge
					if(tid == 0) // thread is not going to update R_cct[tid]...
					{
						cur.R_cct[cur_vertices] = new_edges;
						// ...so special cases are handled like this
						cur.R_cct[0] = 0;
						Unmerge(uv[new_edges - 1], p, q);
						while(++p < cur_vertices)
						{
							cur.R_cct[p] = new_edges;
						}
					}
					else
					{
						Vertex r; // The vertex at the previous position
						Unmerge(uv[tid - 1], r, q);
						// r is obtained from unmerge. q is not needed
						if(r == p - 1)
							cur.R_cct[p] = tid;
						// We sorted the UCM data. So if graph is connected
						// either r = p, or r = p-1. Nothing to do if r=p.
						// Else if r = p-1, we update position for R_cct
						else if(p != r) // If this is true, then there are some
										// super-vertices that are not connected
										// with any other super-vertex
						{
							while(r < p)
								cur.R_cct[++r] = tid;
							// Set the R_cct accordingly for now.
						}
					}
				}
			}
			struct PreallocatedMemoryUnit
			{
				Vertex *VBytes;
				Weight *WBytes;
				Weight *XBytes;
				GraphEdge *ResultStart, *ResultEnd;
				CMST_Kernels::UncompressedEdge HCopy;
				cudaStream_t s0, s1, s2, s3;
			};
		} // namespace CMST_Kernels

		class MST_CUDA_Solver
		{
		   private:
			MST_Result MST_CUDA(SparseGraph* graph_host, 
				CMST_Kernels::PreallocatedMemoryUnit* unit);
			std::string VerifyUnit(Vertex vertices, 
				const CMST_Kernels::PreallocatedMemoryUnit* unit) const
			{
				if(unit->VBytes == nullptr)
					return "VBytes cannot be null!";
				else if(unit->WBytes == nullptr)
					return "WBytes cannot be null!";
				else if(unit->XBytes == nullptr)
					return "XBytes cannot be null!";
				else if(unit->ResultStart == nullptr)
					return "ResultStart cannot be null!";
				else if(unit->ResultEnd == nullptr)
					return "ResultEnd cannot be null!";
				else if(unit->ResultEnd - unit->ResultStart < vertices)
					return "Insufficient memory to store result";
				else if(unit->HCopy.uv == nullptr || unit->HCopy.w == nullptr)
					return "HCopy is not initialized!";
				else if(unit->s0 == nullptr ||
					unit->s1 == nullptr ||
					unit->s2 == nullptr ||
					unit->s3 == nullptr)
					return "streams is not initialized!";
				return "";
			}
			MST_Result MST_CUDA_Update(GraphEdge* old_edges, Weight old_edge_count, 
							Vertex old_vertices, GraphEdge* new_edges,
							Weight new_edge_count, Vertex new_vertices,
							bool free_Memory=false);
			void InitializeUnit(CMST_Kernels::PreallocatedMemoryUnit& unit,
				Vertex vertices, Weight edge_count)
			{
				Weight wb, vb, xb, tb, available, total;
				tb = GPUBytesRequired(vertices, edge_count, wb, vb, xb);
				cudaMemGetInfo(&available, &total);
				if(tb == 0)
				{
					std::cout
						<< "\nOverflow/wrap while allocating memory! Terminating ...\n";
					throw unit;
				}

				#if DEBUG
				std::cout << "\nwb: " << wb << "\nvb: " << vb;
				std::cout << "\nxb: " << xb << "\ntb: " << tb;
				#endif

				std::cout << "\nGPU Capacity = ";
				PrintByteSize(total);
				std::cout << "\nPreparing to allocate ";
				PrintByteSize(tb);
				std::cout << " in GPU memory...\nAvailable memory = ";
				PrintByteSize(available);
				std::cout << "\n";

				if(available <= tb)
				{
					std::cout
						<< "\nInsufficient memory available! Terminating ...\n";
					throw available;
				}
				
				gpuErrchk(cudaStreamCreate(&unit.s0));
				gpuErrchk(cudaStreamCreate(&unit.s1));
				gpuErrchk(cudaStreamCreate(&unit.s2));
				gpuErrchk(cudaStreamCreate(&unit.s3));

				gpuErrchk(cudaMalloc(&unit.WBytes, wb));
				gpuErrchk(cudaMalloc(&unit.VBytes, vb));
				gpuErrchk(cudaMalloc(&unit.XBytes, xb));
				
				unit.HCopy.uv = MALLOC<Weight>(vertices);
				unit.HCopy.w = MALLOC<Weight>(vertices);
			}
			void FreeUnit(CMST_Kernels::PreallocatedMemoryUnit& unit)
			{
				gpuErrchk(cudaStreamDestroy(unit.s0));
				gpuErrchk(cudaStreamDestroy(unit.s1));
				gpuErrchk(cudaStreamDestroy(unit.s2));
				gpuErrchk(cudaStreamDestroy(unit.s3));

				cudaFree(unit.WBytes);
				cudaFree(unit.VBytes);
				cudaFree(unit.XBytes);

				free(unit.HCopy.uv);
				free(unit.HCopy.w);
			}
			Weight LoadEdges(GraphEdge* edge, const Weight atMost, 
				AbstractGraphStreamReader* stream)
			{
				Weight loaded = 0;
				std::cout << "at most weight " << atMost<<std::endl;
				while(atMost > loaded && stream->EdgesRemainaing())
				{
					edge[loaded++] = stream->ReadNextEdge();
				}
				return loaded;
			}
			Weight LoadEdgesFromArray(GraphEdge* edge, const Weight atMost,
				CustomArray::Array1D<GraphEdge>* graphEdge1D, Weight total_edges, Weight &current_index)
			{
				Weight loaded = 0;
				while(atMost>loaded && current_index<total_edges)
				{
					edge[loaded++] = graphEdge1D->Get(current_index++);
				}
				return loaded;
			}
		   public:
			size_t GPUBytesRequired(const Vertex vertices, 
				const Weight edges,
				Weight& wbytes, Weight& vbytes, Weight& xbytes)
			{
				// Total size allocated =
				//		(m * 2 * 3) * sizeof(Weight) +
				//		(((2m + n + 1) * 2 ) + n) * sizeof(Vertex) +
				// 		MAX ((m * 4) * sizeof(Weight), (v * sizeof(Vertex)))
				//		Where n = vertices, m = edges;
				wbytes = vbytes = xbytes = 0;
				Weight s1, s2, s3, s4;
				s1 = SafeMultiplication(edges, 6 * sizeof(Weight));
				s2 = SafeMultiplication(edges, 2); // de
				s3 = SafeAddition(vertices, 1); // vp1
				if(s1 == 0 || s2 == 0 || s3 == 0) return 0;
				wbytes = s1;
				s2 = SafeAddition(s2, s3); // vp1 + de
				s2 = SafeMultiplication(s2, 2); // 2*(vp1+de)
				if(s2 == 0) return 0;
				s2 = SafeAddition(s2, vertices); // v + 2*(vp1+de)
				s2 = SafeMultiplication(s2, sizeof(Vertex));
				if(s2 == 0) return 0;
				vbytes = s2;
				s3 = SafeMultiplication(edges, 4 * sizeof(Weight));
				s4 = SafeMultiplication(vertices, sizeof(Vertex));
				if(s3 == 0 || s4 == 0) return 0;
				xbytes = s3 > s4 ? s3: s4;
				s3 = xbytes;
				s2 = SafeAddition(s2, s3);
				if(s2 == 0) return 0;
				s1 = SafeAddition(s1, s2);
				return s1;
			}
			MST_Result MST_CUDA(SparseGraph* graph_host)
			{
				return MST_CUDA(graph_host, nullptr);
			}
			MST_Result MST_CUDA_Streamed(const Vertex vertices,
							AbstractGraphStreamReader* stream,
							const Weight bufferEdgeCount,
							GraphEdge* bufferArea=nullptr);
			// begin new chunk
			MST_Result MST_CUDA_Streamed_New(const Vertex vertices,
							AbstractGraphStreamReader* stream,
							const Weight bufferEdgeCount,
							GraphEdge* bufferArea=nullptr,
							SparseGraph** graphArray=nullptr,
							CustomArray::Array1D<GraphEdge>* graphEdge1d=nullptr,
							int numChunks=1);
			//end new chunk
		};

		/**
		 * @brief Updates the MST, after taking into account
		 * the newly added edges.
		 * UNSAFE WARNING if `free_Memory=true`: frees all 
		 * allocated edges (`old_edges` and `new_edges`)
		 * 
		 * @param old_edges The array of old edges
		 * @param old_edge_count The count of edges in the old MST
		 * @param old_vertices The count of vertices in the old MST
		 * @param new_edges The array of new/updated edgelist
		 * @param new_edge_count The count of new edges 
		 * @param new_vertices The number of vertices in the new graph.
		 * @param free_Memory If true, memory is managed optimally and freed
		 * after usage. If false, memory is copied and left unchanged.
		 * Cannot be less than `old_vertices`
		 * @return MST_Result 
		 */
		MST_Result MST_CUDA_Solver::MST_CUDA_Update(GraphEdge* old_edges, Weight old_edge_count, 
									Vertex old_vertices, GraphEdge* new_edges, 
									Weight new_edge_count, Vertex new_vertices,
									bool free_Memory)
		{
			if(new_vertices < old_vertices || old_edges == nullptr || new_edges == nullptr)
				throw;
			GraphEdge* merged;
			size_t full_size = SafeAddition(old_edge_count, new_edge_count);
			full_size = SafeMultiplication(full_size, sizeof(GraphEdge));
			if(free_Memory && full_size != 0)
			{
				if(new_edge_count > old_edge_count)
				{ // Swap edges
					// Swap (old_vertices, new_vertices)
					Vertex tv = old_vertices;
					old_vertices = new_vertices;
					new_vertices = tv;

					// Swap (old_edge_count, new_edge_count)
					Weight tw = old_edge_count;
					old_edge_count = new_edge_count;
					new_edge_count = tw;

					// Swap (old_edges, new_edges)
					GraphEdge *tp = old_edges;
					old_edges = new_edges;
					new_edges = tp;
				}
				merged = (GraphEdge*) realloc(
					old_edges, 
					full_size);
				if(merged == nullptr)
					throw "Memory allocation failed!";
				memcpy(
					merged + old_edge_count, 
					new_edges, 
					new_edge_count * sizeof(GraphEdge));
				free(new_edges);
				new_edges = nullptr;
			}
			else
			{
				merged = MALLOC<GraphEdge>(new_edge_count + old_edge_count);
				if(merged == nullptr)
					throw "Memory allocation failed!";
				memcpy(
					merged, old_edges,
					sizeof(GraphEdge) * old_edge_count);
				memcpy(
					merged + old_edge_count, new_edges, 
					sizeof(GraphEdge) * new_edge_count);
			}
			CustomArray::Array1D<GraphEdge> merged_array(
				merged, new_edge_count+old_edge_count, false);
			auto graph = new SparseGraph(
				merged_array, new_vertices, false); 
			auto result =  MST_CUDA(graph);
			delete graph;
			if (free_Memory)
			{
				if(merged != nullptr)
					free(merged);
				if(old_edges != nullptr)
					free(old_edges);
				if(new_edges != nullptr)
					free(new_edges);
				merged = old_edges = new_edges = nullptr;
			}
			return result;
		}

		/**
		 * @brief CUDA algorithm for finding the MST of a
		 * given graph using Boruvka approach
		 * 
		 * This is an internal function and therefore is marked as static
		 * @param graph_host: The graph defined in the host memory
		 * @param unit The preallocated stuct. The `MST_CUDA_Streamed` function
		 * makes use of this custom defined struct to avoid reinitialization of
		 * of memory. 
		 * @return CustomArray::Array1D<GraphEdge> The list of edges selected in
		 * the MST
		 */
		MST_Result MST_CUDA_Solver::MST_CUDA(SparseGraph* graph_host, 
			CMST_Kernels::PreallocatedMemoryUnit* unit)
		{
			MST_Result mst_result;
			mst_result.Edge_Count = 0;
			mst_result.Components = 0;
			Vertex vertices		  = graph_host->get_Vertices();
			#if DEBUG
			std::cout << "Beginning the new CUDA MST Algorithm...\n";
			clock_t			  a, b;
			Vertex			  itr_count = 0;
			Custom_CUDA_timer timer;
			float			  tms, fulltime = 0;
			float			  kernel_times[10] = {0};
			#endif
			mst_result.EdgeList = nullptr;
			Vertex					  edges, i;
			CMST_Kernels::CSR_Wrapper old, cur;
			const Vertex			  blocksize = BLOCKSIZE;
			Vertex		 gridsize = (vertices + blocksize - 1) / blocksize;
			cudaStream_t parallel_stream[4]; // Parallel stream for CUDA
			CMST_Kernels::AdjListNode	   Nearest;
			CMST_Kernels::UncompressedEdge UCM, UCopy, h_Copy;
			Weight	de; // Directed edges = 2 * the number of undirected edges
			Weight* AllHostWeightArray;
			Weight* AllXByteArray;
			// Creating one unit of memory for type 'Weight'
			Vertex* AllHostVertexArray;
			GraphEdge* result_array;
			// Creating one unit of memory for type 'Vertex'
			thrust::device_ptr<Vertex> Nearest_ptr;
			thrust::device_ptr<Weight> UCM_uv_ptr, UCM_w_ptr, UCopy_uv_ptr,
				UCopy_w_ptr, cur_nnz_ptr, nst_wt_ptr;

			{										  // Initialization phase
				edges = graph_host->get_Edge_Count(); // Number of edges
				if(edges < 1 || vertices < 2 || graph_host->isDirected())
				{
					mst_result.Components = vertices;
					mst_result.Total_Cost = 0;
					return mst_result;
					return mst_result;
				}
				de = edges + edges;
				// number of directed edges in the adjoint matrix
				Vertex vp1 = vertices + 1;
				// Number of elements in a CSR matrix's R_cct array
				std::string message;
				if(unit == nullptr)
				{
					#if DEBUG
					std::cout << "\n@VER:" << vertices << "\n@EDG:" << edges;
					for(i = 0; i < 10; i++)
					{
						kernel_times[i] = 0;
					}
					std::cout << "\nInitialize memory on device...\n";
					a = clock();
					#endif

					Weight WBytes, XBytes, available, total, VBytes;
					double TotalBytes = this->GPUBytesRequired(vertices, edges,
						WBytes, VBytes, XBytes);
					cudaMemGetInfo(&available, &total);
					if(TotalBytes == 0)
					{
						std::cout
							<< "\nOverflow/wrap while allocating memory! Terminating ...\n";
						throw graph_host;
					}

					#if DEBUG
					std::cout << "\nwb: " << WBytes << "\nvb: " << VBytes;
					std::cout << "\nxb: " << XBytes << "\ntb: " << TotalBytes;
					#endif

					std::cout << "\nGPU Capacity = ";
					PrintByteSize(total);
					std::cout << "\nPreparing to allocate ";
					PrintByteSize(TotalBytes);
					std::cout << " in GPU memory...\nAvailable memory = ";
					PrintByteSize(available);
					std::cout << "\n";

					if((double)available <= TotalBytes)
					{
						std::cout
							<< "\nInsufficient memory available! Terminating ...\n";
						throw available;
					}

					gpuErrchk(cudaMalloc(&AllHostWeightArray, WBytes));
					gpuErrchk(cudaMalloc(&AllHostVertexArray, VBytes));
					gpuErrchk(cudaMalloc(&AllXByteArray, XBytes));
					result_array = MALLOC<GraphEdge>(vertices - 1);
					h_Copy.uv  = MALLOC<Weight>(vertices);
					// Prepare Host Nearest node-pair array
					h_Copy.w = MALLOC<Weight>(vertices);
				}
				else if ((message = this->VerifyUnit(vertices, unit)) != "")
				{
					std::cout << "ERROR!\n" << message <<"\n";
					throw unit;
				}
				else
				{
					AllHostVertexArray = unit->VBytes;
					AllHostWeightArray = unit->WBytes;
					AllXByteArray = unit->XBytes;
					result_array = unit->ResultStart;
					h_Copy.uv = unit->HCopy.uv;
					h_Copy.w = unit->HCopy.w;
					parallel_stream[0] = unit->s0;
					parallel_stream[1] = unit->s1;
					parallel_stream[2] = unit->s2;
					parallel_stream[3] = unit->s3;
				}
				Nearest.weight = AllHostWeightArray;
				cur.NNZ = Nearest.weight + de; // size of Nearest.weight = de
				old.NNZ = cur.NNZ + de;		   // size of cur.NNZ = de
				// size of old.NNZ = de

				UCM.w  = AllXByteArray;
				UCM.uv = UCM.w + de; // size of UCM.w = de

				cur.R_cct	   = AllHostVertexArray;
				old.R_cct	   = cur.R_cct + vp1; // size of cur.R_cctx = vp1
				cur.C_idx	   = old.R_cct + vp1; // size of old.R_cct = vp1
				old.C_idx	   = cur.C_idx + de;  // size of cur.C_idx = vp1
				Nearest.node   = old.C_idx + de;  // size of old.C_idx = de
				Nearest.origin = (Vertex*)UCM.w;
				UCopy.uv	   = (Weight*)cur.C_idx;
				UCopy.w		   = old.NNZ;
				// size of UCopy.weight = de = size of Nearest.weight.
				// sizeof Nearest.node = vertices

				// Total = (vp1 * 2) + (de * 2) + (vertices * 2)
				Nearest_ptr	 = thrust::device_ptr<Vertex>(Nearest.node);
				UCM_uv_ptr	 = thrust::device_ptr<Weight>(UCM.uv);
				UCM_w_ptr	 = thrust::device_ptr<Weight>(UCM.w);
				UCopy_uv_ptr = thrust::device_ptr<Weight>(UCopy.uv);
				UCopy_w_ptr	 = thrust::device_ptr<Weight>(UCopy.w);
				cur_nnz_ptr	 = thrust::device_ptr<Weight>(cur.NNZ);
				nst_wt_ptr	 = thrust::device_ptr<Weight>(Nearest.weight);
			}
			{ // Buildup phase
				if(unit == nullptr) gpuErrchk(cudaStreamCreate(&parallel_stream[1]));
				// Initializing parallel stream for copying row Cumulative count

				Vertex vpb = (vertices + 1) * sizeof(Vertex);
				// calculate the size needed beforehand
				gpuErrchk(cudaMemcpyAsync(old.R_cct,
										  &(graph_host->AdjMat.row_index[0]),
										  vpb, cudaMemcpyHostToDevice,
										  parallel_stream[1])); // copy elements

				
				if(unit == nullptr) gpuErrchk(cudaStreamCreate(&parallel_stream[2]));
				// Initialize a parallel stream for copying column indices
				Vertex evb = edges * sizeof(Vertex);
				// bytes needed for initialization  is calculated beforehand
				gpuErrchk(cudaMemcpyAsync(
					old.C_idx, &(graph_host->AdjMat.col_index[0]), evb,
					cudaMemcpyHostToDevice,
					parallel_stream[2])); // Copying evb bytes only

				
				if(unit == nullptr) gpuErrchk(cudaStreamCreate(&parallel_stream[3]));
				// Initialize a parallel stream for copying the weight
				Vertex eb = edges * sizeof(Weight);
				// bytes needed for initialization is calculated beforehand
				gpuErrchk(cudaMemcpyAsync(
					old.NNZ, &(graph_host->AdjMat.NNZ[0]), eb,
					cudaMemcpyHostToDevice,
					parallel_stream[3])); // copying eb bytes only

				cudaDeviceSynchronize(); // Wait for all the streams to end
				
				if(unit == nullptr) gpuErrchk(cudaStreamCreate(&parallel_stream[0]));
				// Create a parallel stream for this task

				#if DEBUG
				timer.StartRecord();
				#endif

				CMST_Kernels::getInitEdgeList<<<gridsize, blocksize>>>(
					vertices, edges, old, UCM, cur);
				// Preparing unordered edge list

				cudaDeviceSynchronize(); // Wait for edgelist to be prepared
										 #if DEBUG
				tms = timer.StopRecord();
				fulltime += tms;
				kernel_times[0] = tms;
				timer.StartRecord();
				#endif
				thrust::sort_by_key(UCM_uv_ptr, UCM_uv_ptr + de, cur_nnz_ptr);
				gridsize = (de + blocksize - 1) / blocksize;
				CMST_Kernels::getInitCsrMatrix<<<gridsize, blocksize, 0,
												 parallel_stream[0]>>>(
					vertices, de, UCM, cur);
				// Prepare the new symmetric matrix in `cur`
				#if DEBUG
				cudaDeviceSynchronize();
				tms = timer.StopRecord();
				fulltime += tms;
				kernel_times[1] = tms;
				#endif

				gridsize   = (vertices + blocksize - 1) / blocksize;
				// Prepare Host Nearest.weight array
			}
			{ // Iteration phase
				mst_result.Total_Cost = 0; // Cost is 0 initially
				std::cout << "\nNow beginning iteration phase...\n";
				int itr_count=0;
				while(true)
				// Starting Boruvka approach
				{
					itr_count++;
					std::cout << "\nIteration " << itr_count << "...\n";
					cudaDeviceSynchronize();
					// Waiting for `Init_CSR_Matrix()`, or `NewEdgeList_k2`

					#if DEBUG
					b			= clock();
					double time = 1000 * ((double)b - a) / CLOCKS_PER_SEC;
					if(itr_count++ == 0)
						std::cout << "\n#S0 " << time << " ms for Setup.\n";
					else
						std::cout
							<< "\n#In " << time
							<< " ms for this iteration.\nIteration vertices = "
							<< vertices
							<< "; Time taken by previous iteration = " << time
							<< "; Worst case ETA: " << (time * 2) << "\n";
					a = clock();
					timer.StartRecord();
					#endif
					std::cout << "\n starting find sucessor\n";
					CMST_Kernels::FindSuccessor<<<gridsize, blocksize, 0,
												  parallel_stream[0]>>>(
						vertices, cur, Nearest, UCM);
					// Get successor, obtain selected edge_ids in UCM.w

					std::cout << "\n starting memcpy\n";
					cudaMemcpyAsync(old.NNZ, cur.NNZ, de * sizeof(Weight),
									cudaMemcpyDeviceToDevice,
									parallel_stream[3]);
					// Overwritting the old CSR matrix in a parallel stream
					cudaMemcpyAsync(old.C_idx, cur.C_idx, de * sizeof(Vertex),
									cudaMemcpyDeviceToDevice,
									parallel_stream[1]);
					// Overwritting the old CSR matrix in a parallel stream
					cudaMemcpyAsync(
						old.R_cct, cur.R_cct, (vertices + 1) * sizeof(Vertex),
						cudaMemcpyDeviceToDevice, parallel_stream[2]);
					std::cout << "\nfinished memcpy\n";
					// Overwritting the old CSR matrix in a parallel stream
					cudaStreamSynchronize(parallel_stream[0]);
					cudaStreamSynchronize(parallel_stream[3]);
					// Wait for the `FindSuccessor()` kernel
					#if DEBUG
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[2] += tms;
					timer.StartRecord();
					#endif
					CMST_Kernels::RemoveCycles<<<gridsize, blocksize, 0,
												 parallel_stream[0]>>>(
						vertices, Nearest, cur.NNZ);
					std::cout << "\n finished remove cycles\n";
					// Remove cycles, copy edges from UCM to UCopy
					#if DEBUG
					cudaDeviceSynchronize();
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[3] += tms;
					#endif
					cudaStreamSynchronize(parallel_stream[0]);
					// Wait for the `RemoveCycle()` kernel to finish
					#if DEBUG
					tms = timer.StopRecord();
					fulltime += tms;
					#endif
					std::cout << "start memcpy to host \n";
					cudaMemcpyAsync(
						h_Copy.uv, UCM.w, sizeof(Weight) * vertices,
						cudaMemcpyDeviceToHost,
						parallel_stream[1]); // copy values to host from device
					cudaMemcpyAsync(
						h_Copy.w, cur.NNZ, sizeof(Weight) * vertices,
						cudaMemcpyDeviceToHost,
						parallel_stream[2]); // copy values to host from device
					cudaMemcpyAsync(Nearest.weight, UCM.uv, sizeof(Weight) * de,
									cudaMemcpyDeviceToDevice,
									parallel_stream[3]);
					std::cout << "finished memcpy to host \n";
					// copy Edge-Ids to the now unused array Nearest.Weight
					#if DEBUG
					cudaDeviceSynchronize();
					timer.StartRecord();
					#endif
					std::cout << "start update root kernel\n";
					CMST_Kernels::UpdateRoot<<<gridsize, blocksize, 0,
											   parallel_stream[0]>>>(
						vertices, Nearest.node, cur.R_cct);
					cudaDeviceSynchronize();
					std::cout << "finished update root kernel\n";
					// Find Root for each sub-tree in the forest.
					#if DEBUG
					cudaDeviceSynchronize();
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[4] += tms;
					#endif
					cudaStreamSynchronize(parallel_stream[1]);
					// Wait for copy nearest.node task
					// to be over
					cudaStreamSynchronize(parallel_stream[2]);
					// Wait for copy nearest.pair task
					// to be over
					cudaDeviceSynchronize(); // Wait for all threads to stop.
					bool zero_degree = false; // 0-degrees found this time
					std::cout << "starting the edges adding logic \n";
					for(i = 0; i < vertices; i++)
					// As the last kernel runs, add edges to the solution
					{
						Weight x = h_Copy.uv[i], y = h_Copy.w[i];
						Vertex p, q;
						// The vertices that can be obtained from the merged uv
						// pair
						CMST_Kernels::Unmerge(x, p, q);
						// Get original vertices from the edge_id
						if(x == CMST_Kernels::invalidW)
						// if Some node is found with no neighbour...
						{
							mst_result.Components++;
							zero_degree = true;
							// A forest is found.
						}
						else if(y != 0) // If this is not a root node...
						{
							// Add this edge to list of edges is solution
							result_array[mst_result.Edge_Count++] = p < q ?
								GraphEdge(p, q, y):
								GraphEdge(q, p, y);
								//Obtained the correct edges
								//std::cout << p << " " << q << " " << y << std::endl;
							// ...then add this edge to the MST list of edges
							mst_result.Total_Cost += y;
						}
					}
					std::cout << "end of the edges adding logic \n";
					cudaDeviceSynchronize(); // Wait for all threads to stop.

					//Checkpoint works here too.
					//for(int i=0; i<mst_result.Edge_Count; i++){
					//	GraphEdge e = result_array[i];
					//	std::cout << e.u << " " << e.v << " " << e.w << std::endl;
					//}

					if(mst_result.Graph_Vertices() >=
					   graph_host->get_Vertices())
						break;
					std::cout << "wait for all threads to stop\n";
					cudaDeviceSynchronize(); // Wait for all threads to stop.
					std::cout << "devicetodevice memcpy\n";
					cudaMemcpy(Nearest.node, cur.R_cct,
							   sizeof(Vertex) * vertices,
							   cudaMemcpyDeviceToDevice);
					// Copy roots of every subtree to `Nearest.node`
					thrust::sort(Nearest_ptr, Nearest_ptr + vertices);
					std::cout << "done devicetodevice memcpy\n";

					// 0-degree vertices now at the end; so they get discarded
					// Sort roots, every valid value now repeats at least twice
					auto end =
						thrust::unique(Nearest_ptr, Nearest_ptr + vertices);
					// We are left with all unique roots, and the
					// cluster/supervertex id can be taken as the
					// position of this root in this array.
					Vertex cur_vertices = end - Nearest_ptr;
					std::cout << "cur_vertices: " << cur_vertices << std::endl;
					if(zero_degree) // All zero-degree vertices are clubbed together
						cur_vertices--; // They are not to be considered in the next graph 
					// The number of unique elements in the root array
					#if DEBUG
					std::cout
						<< "Disconnected components: " << mst_result.Components
						<< ", Found " << cur_vertices
						<< " clusters i.e super vertices.\n";
					#endif
					if(cur_vertices == 1) // We're done here
					{
						mst_result.Components++;
						// No need to merge further, we know the answer
						break; // Exit the loop
					}
					#if DEBUG
					timer.StartRecord();
					#endif
					CMST_Kernels::CreateIndexMapper<<<
						(cur_vertices + blocksize - 1) / blocksize,
						blocksize>>>(cur_vertices, Nearest.node,
									 Nearest.origin);
					std::cout << "done create index mapper\n";

					cudaDeviceSynchronize();
					// Wait for ID assignment to be  completed
					#if DEBUG
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[5] += tms;
					timer.StartRecord();
					#endif
					CMST_Kernels::MapRootToIndex<<<gridsize, blocksize>>>(
						vertices, Nearest.origin, cur.R_cct);
					std::cout << "done map root to index\n";

					cudaDeviceSynchronize();
					// Wait for ID assignment to be  completed
					#if DEBUG
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[6] += tms;
					timer.StartRecord();
					#endif

					CMST_Kernels::NewEdgeList_k1<<<gridsize, blocksize>>>(
						vertices, cur_vertices, old, cur.R_cct, UCM);
					std::cout << "done new edge list k1\n";
					// Prepare new uncompressed edge list

					auto zipped_input =
							 thrust::make_zip_iterator(thrust::make_tuple(
								 UCM_w_ptr,
								 nst_wt_ptr)), // Create zip iterators to...
						zipped_output = thrust::make_zip_iterator(
							thrust::make_tuple(UCopy_w_ptr, cur_nnz_ptr));
					std::cout << "done make zip iterator\n";
					//...maintain the order of elements

					cudaDeviceSynchronize();
					std::cout << "wait for all threads to stop\n";
					// Wait for uncompressed edge list to be generated

					#if DEBUG
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[7] += tms;
					#endif
					std::cout << "starting sort by key\n";

					//SORT BY KEY FAILURE

					std::cout << "UCM_uv_ptr: " << UCM_uv_ptr << std::endl;
					std::cout << "end of ucm "<< UCM_uv_ptr+ edges+edges << std::endl;
					std::cout << thrust::get<0>(zipped_input[0]);

					thrust::sort_by_key(UCM_uv_ptr, UCM_uv_ptr + edges + edges,
										zipped_input);
					std::cout << "done sort by key\n";
					// Sort uv pairs. Here uv pair are clusters/super-vertices
					auto result =
						thrust::reduce_by_key // Reduce wrt the same uv key
						(UCM_uv_ptr, UCM_uv_ptr + edges + edges,
						 // Start and end of input iteration
						 zipped_input, UCopy_uv_ptr, zipped_output,
						 thrust::equal_to<Weight>(), CMST_Kernels::Min_op());
					std::cout << "done reduce by key\n";

					Vertex cur_edges = result.first - UCopy_uv_ptr - 1;
					// subtracted 1 for all the clubbed invalid uv pair
					if(cur_edges == 0) // No edges, but multiple vertices remain
					{
						// Update the number of disconnected components
						mst_result.Components += cur_vertices;

						break; // Result is found. Stop the loop
					}
					std::cout << "starting memcpy\n";
					cudaMemcpy(Nearest.weight, UCopy.uv,
							   (edges + edges) * sizeof(Weight),
							   cudaMemcpyDeviceToDevice);
					gridsize = (cur_edges + blocksize - 1) / blocksize;
					// gridsize for (cur_edges) threads
					std::cout << "Cur edges: " << cur_edges << std::endl;

					#if DEBUG
					if((cur_edges & 1) > 0)
					{
						std::cout << "WARNING! EDGES is not even!\n\n";
					}
					timer.StartRecord();
					#endif
					CMST_Kernels::NewEdgeList_k2<<<gridsize, blocksize, 0,
												   parallel_stream[0]>>>(
						cur_edges, UCopy, Nearest.weight, cur_vertices, cur,
						UCM);

					#if DEBUG
					cudaDeviceSynchronize();
					tms = timer.StopRecord();
					fulltime += tms;
					kernel_times[8] += tms;
					#endif

					// Number of vertices changed, so gridsize is updated.
					vertices = cur_vertices;
					de		 = cur_edges;
					edges	 = cur_edges / 2;
					// Edge are repeated, since this is a symmetric csr matrix
					gridsize = (vertices + blocksize - 1) / blocksize;
				}
				if(unit == nullptr)
				{ // Reallocate (if needed) and Initialize the array in mst_result
					//if(mst_result.Components > 1)
					//{
					//	auto * act_array = (GraphEdge*) realloc(result_array, 
					//		mst_result.Edge_Count * sizeof(vertices));
					//	if(act_array != nullptr)
					//		result_array = act_array;
					//}
					mst_result.EdgeList =
						new CustomArray::Array1D<GraphEdge>(result_array, 
							mst_result.Edge_Count, false);
				}
				else
					mst_result.EdgeList = nullptr; // Only deal with raw array in *unit
			}
			{ // Memory cleanup phase
				cudaDeviceSynchronize();
				// Stop all GPU threads (if any exceptions are thrown)
				std::cout << "\nAlgorithm complete. Now clearing memory ...\n";
				// Destroying the parallel streams
				if(unit == nullptr)
				{
					cudaStreamDestroy(parallel_stream[0]);
					cudaStreamDestroy(parallel_stream[1]);
					cudaStreamDestroy(parallel_stream[2]);
					cudaStreamDestroy(parallel_stream[3]);
					// Clearing memory on the GPU
					cudaFree(AllHostWeightArray);
					cudaFree(AllHostVertexArray);
					cudaFree(AllXByteArray);
					// Clearing extra memory on the host
					free(h_Copy.uv);
					free(h_Copy.w);
				}
				else
				{
					parallel_stream[0] = nullptr;
					parallel_stream[1] = nullptr;
					parallel_stream[2] = nullptr;
					parallel_stream[3] = nullptr;
				}
				#if DEBUG
				kernel_times[9] = fulltime;
				if(graph_host->get_Vertices() == mst_result.Graph_Vertices())
				{
					std::cout << "Components are numbered correctly!";
				}
				else
				{
					std::cout << "WARNING: Components are incorrect!!";
				}
				for(i = 0; i < 10; i++)
				{
					std::cout << "\n@KT" << i << ":" << kernel_times[i];
				}
				std::cout << "\n\n";
				#endif
			}
			return mst_result;
		}

		/**
		 * @brief Runs MST on a stream of graph edges, with a given buffer size
		 * 
		 * @param vertices The (maximum) number of vertices expected 
		 * @param stream The stream of edges to read
		 * @param bufferEdgeCount The number of edges in buffer. Must be 
		 * greater than or equal to number of vertices
		 * @param bufferArea The Preallocated memory for buffer (optional,
		 * can be left as nullptr)
		 * @return MST_Result Returns the MST for the graph read in the stream
		 */
		MST_Result MST_CUDA_Solver::MST_CUDA_Streamed(const Vertex vertices,
						AbstractGraphStreamReader* stream,
						const Weight bufferEdgeCount,
						GraphEdge* bufferArea)
		{
			if (bufferEdgeCount < vertices)
				throw "Cannot have a edge buffer count less than the number of vertices";
			CMST_Kernels::PreallocatedMemoryUnit unit;
			MST_Result result;
			InitializeUnit(unit, vertices, bufferEdgeCount);
			{ // MST phase
				GraphEdge* processing; // Make space for buffer
				processing = bufferArea == nullptr? // is the provided memory invalid?
					MALLOC<GraphEdge>(bufferEdgeCount): // Allocate memory for buffer
					bufferArea; // Make use of this buffer
				if(processing == nullptr) throw processing; // Check allocation is valid
				CustomArray::Array1D<GraphEdge>* current_buffer; // Prepare a pointer for 1D array
				Weight loaded = LoadEdges(processing, bufferEdgeCount, stream); // Load buffer
				std::cout << "initial load " << loaded << " edges\n";
				std::cout << "buffer edgecount "<< bufferEdgeCount << std::endl;
				current_buffer = new CustomArray::Array1D<GraphEdge>(
					processing, loaded, false); // wrap the edges in an array
				SparseGraph *graph_host = new SparseGraph(*current_buffer, 
					vertices, false); // Prepare a graph out of the array
				unit.ResultStart = processing; // We will overwrite the buffer array (partially) with result
				unit.ResultEnd = processing + vertices; // This is why (bufferEdgeCount >= vertices) is needed
				do
				{
					result = this->MST_CUDA(graph_host, &unit);
					if(stream->EdgesRemainaing() == false)
						break;
					loaded = result.Edge_Count +
						LoadEdges(processing + result.Edge_Count, 
							bufferEdgeCount - result.Edge_Count, stream);
					std::cout << "loaded " << loaded << " edges\n";
					std::cout << "edge count result" << result.Edge_Count<< "\n";
					std::cout << "graph host edges "<< graph_host->get_Edge_Count() << std::endl;
					current_buffer = new CustomArray::Array1D<GraphEdge>(processing, 
						loaded, false);
					graph_host->Unsafe_Overwrite(*current_buffer, vertices);
				}while(true);
				// Free memory, copy result
				delete graph_host;
				result.EdgeList = new CustomArray::Array1D<GraphEdge>(unit.ResultStart, 
					unit.ResultEnd - unit.ResultStart, true);
				if(bufferArea == nullptr)
					free(processing);
			}
			FreeUnit(unit);
			return result;
		}


		//BEGIN IMPROVED CHUNK
		/**
		 * @brief Runs MST on a stream of graph edges, with a given buffer size
		 * 
		 * @param vertices The (maximum) number of vertices expected 
		 * @param stream The stream of edges to read
		 * @param bufferEdgeCount The number of edges in buffer. Must be 
		 * greater than or equal to number of vertices
		 * @param bufferArea The Preallocated memory for buffer (optional,
		 * can be left as nullptr)
		 * @param graph: The graph defined in host memory
		 * @return MST_Result Returns the MST for the graph read in the stream
		 */
		MST_Result MST_CUDA_Solver::MST_CUDA_Streamed_New(const Vertex vertices,
						AbstractGraphStreamReader* stream,
						const Weight bufferEdgeCount,
						GraphEdge* bufferArea,
						SparseGraph** graphArray,
						CustomArray::Array1D<GraphEdge>* graphEdge1d,
						int numChunks)
		{
			if (bufferEdgeCount < vertices)
				throw "Cannot have a edge buffer count less than the number of vertices";
			CMST_Kernels::PreallocatedMemoryUnit unit;
			MST_Result result;
			InitializeUnit(unit, vertices, bufferEdgeCount);
			{ // MST phase
				GraphEdge* processing; // Make space for buffer
				processing = bufferArea == nullptr? // is the provided memory invalid?
					MALLOC<GraphEdge>(bufferEdgeCount): // Allocate memory for buffer
					bufferArea; // Make use of this buffer
				if(processing == nullptr) throw processing; // Check allocation is valid
				CustomArray::Array1D<GraphEdge>* current_buffer; // Prepare a pointer for 1D array
				//Weight loaded = LoadEdgesFromArray(processing, bufferEdgeCount, stream); // Load buffer
				Weight total_edges = graphEdge1d->GetCount();
				std::cout << "tot edges " << total_edges<< std::endl;
				Weight current_index = 0;

   				auto start_transfer = std::chrono::high_resolution_clock::now();
				Weight loaded = LoadEdgesFromArray(processing, bufferEdgeCount, graphEdge1d, total_edges, current_index); // Load buffer
   				auto stop_transfer = std::chrono::high_resolution_clock::now();
				auto duration_transfer = std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer - start_transfer);
				auto start_mst = std::chrono::high_resolution_clock::now();
				auto stop_mst = std::chrono::high_resolution_clock::now();
				auto duration_mst = std::chrono::duration_cast<std::chrono::microseconds>(stop_mst - stop_mst);
				auto start_overwrite = std::chrono::high_resolution_clock::now();
				auto stop_overwrite = std::chrono::high_resolution_clock::now();
				auto duration_overwrite = std::chrono::duration_cast<std::chrono::microseconds>(stop_overwrite - stop_overwrite);
				std::cout << "initial load " << loaded << " edges\n";
				std::cout << "buffer edgecount "<< bufferEdgeCount << std::endl;
				current_buffer = new CustomArray::Array1D<GraphEdge>(
					processing, loaded, false); // wrap the edges in an array
				SparseGraph *graph_host = new SparseGraph(*current_buffer, 
					vertices, false); // Prepare a graph out of the array
				unit.ResultStart = processing; // We will overwrite the buffer array (partially) with result
				unit.ResultEnd = processing + vertices; // This is why (bufferEdgeCount >= vertices) is needed
				do
				{
					start_mst = std::chrono::high_resolution_clock::now();
					result = this->MST_CUDA(graph_host, &unit);
					stop_mst = std::chrono::high_resolution_clock::now();
					duration_mst += std::chrono::duration_cast<std::chrono::microseconds>(stop_mst - start_mst);
					if(current_index >= total_edges)
						break;
					start_transfer = std::chrono::high_resolution_clock::now();
					loaded = result.Edge_Count + LoadEdgesFromArray(processing + result.Edge_Count, 
							bufferEdgeCount - result.Edge_Count, graphEdge1d, total_edges, current_index);
					stop_transfer = std::chrono::high_resolution_clock::now();
					duration_transfer += std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer - start_transfer);
					std::cout << "loaded " << loaded << " edges\n";
					std::cout << "edge count result" << result.Edge_Count<< "\n";
					std::cout << "graph host edges "<< graph_host->get_Edge_Count() << std::endl;
					current_buffer = new CustomArray::Array1D<GraphEdge>(processing, 
						loaded, false);
					start_overwrite = std::chrono::high_resolution_clock::now();
					graph_host->Unsafe_Overwrite(*current_buffer, vertices);
					stop_overwrite = std::chrono::high_resolution_clock::now();
					duration_overwrite += std::chrono::duration_cast<std::chrono::microseconds>(stop_overwrite - start_overwrite);
				}while(true);
				std::cout << "----------------------------------------------------------------\n";
					std::cout << "Total time for transfer: " << duration_transfer.count() << " microseconds\n";
					std::cout << "----------------------------------------------------------------\n";
					std::cout << "Total time for mst: " << duration_mst.count() << " microseconds\n";
					std::cout << "----------------------------------------------------------------\n";
					std::cout << "Total time for overwrite: " << duration_overwrite.count() << " microseconds\n";
					// Free memory, copy result
					delete graph_host;
					result.EdgeList = new CustomArray::Array1D<GraphEdge>(unit.ResultStart, 
						unit.ResultEnd - unit.ResultStart, true);
					if(bufferArea == nullptr)
						free(processing);
				}
				FreeUnit(unit);
				return result;
			}


			//END IMPROVED CHUNK



			/**
			* @brief CUDA algorithm for finding the MST of a
			* given graph using Boruvka approach
			* 
			* This is an internal function and therefore is marked as static
			* @param graph: The graph defined in the host memory
			* @return CustomArray::Array1D<GraphEdge> The list of edges selected in
			* the MST
			*/
			MST_Result MST_CUDA(SparseGraph* graph)
			{
				return MST_CUDA_Solver().MST_CUDA(graph);
			}

			/**
			* @brief Runs MST on a stream of graph edges, with a given buffer size
			* 
			* @param vertices The (maximum) number of vertices expected 
			* @param stream The stream of edges to read
			* @param bufferEdgeCount The number of edges in buffer. Must be 
			* greater than or equal to number of vertices
			* @param bufferArea The Preallocated memory for buffer (optional,
			* can be left as nullptr)
			* @return MST_Result Returns the MST for the graph read in the stream
			*/
			MST_Result MST_CUDA_Streamed(const Vertex vertices,
							AbstractGraphStreamReader* stream,
							const Weight bufferEdgeCount,
							GraphEdge* bufferArea=nullptr)
			{
				MST_Result result;
				result = MST_CUDA_Solver().MST_CUDA_Streamed(vertices, stream, bufferEdgeCount, bufferArea);
				return result;
		}


		//BEGIN IMPROVED CHUNK

		/**
		 * @brief Runs MST on a stream of graph edges, with a given buffer size
		 * 
		 * @param vertices The (maximum) number of vertices expected 
		 * @param stream The stream of edges to read
		 * @param bufferEdgeCount The number of edges in buffer. Must be 
		 * greater than or equal to number of vertices
		 * @param bufferArea The Preallocated memory for buffer (optional,
		 * can be left as nullptr)
		 * @param graph: The graph defined in host memory
		 * @return MST_Result Returns the MST for the graph read in the stream
		 */
		MST_Result MST_CUDA_Streamed_New(const Vertex vertices,
						AbstractGraphStreamReader* stream,
						const Weight bufferEdgeCount,
						GraphEdge* bufferArea=nullptr,
						SparseGraph** graphArray=nullptr,
						CustomArray::Array1D<GraphEdge>* graphEdge1d=nullptr,
						int numChunks=1)
		{
			MST_Result result;
			result = MST_CUDA_Solver().MST_CUDA_Streamed_New(vertices, stream, bufferEdgeCount, bufferArea, graphArray, graphEdge1d, numChunks);
			return result;
		}
		//END IMPROVED CHUNK


		//BEGIN NEW GPU IMPLEMENTATION

		void parallel_gpu_new(SparseGraph** sparseGraphArray, int numChunks, 
								Vertex v, Weight chunkSize, CustomArray::Array1D<GraphEdge> *lastEdgeList, 
								std::string result_file){
		
			Weight total_edges_in_msf_chunks = 0;
			struct Graphs::Algo::MST_Result* msf_gpu_array = (struct Graphs::Algo::MST_Result*)malloc(sizeof(struct Graphs::Algo::MST_Result)*numChunks);




			auto start_parallel = std::chrono::high_resolution_clock::now();

			std::cout << "numChunks: " << numChunks << std::endl;
			std::cout << "chunkSize: " << chunkSize << std::endl;

			for(int i=0; i<numChunks; i++){
				SparseGraph* sparseGraph = sparseGraphArray[i];
				std::cout << "edges " << sparseGraph->get_Edge_Count() << std::endl;
				std::cout << "vertices " << sparseGraph->get_Vertices() << std::endl;
			}
		


			//#pragma omp parallel for
			for(int i=0; i< numChunks; i++){
				msf_gpu_array[i] = Graphs::Algo::MST_CUDA(sparseGraphArray[i]);
			}


			auto stop_parallel= std::chrono::high_resolution_clock::now();
			auto duration_parallel= std::chrono::duration_cast<std::chrono::microseconds>(stop_parallel - start_parallel);
			std::cout << "1st level chunk execution time: " << duration_parallel.count() << " microseconds" << std::endl;

			for(int i=0; i< numChunks; i++){
				total_edges_in_msf_chunks += msf_gpu_array[i].Edge_Count;
			}
			int lastEdgeListSize = lastEdgeList->GetCount();
			total_edges_in_msf_chunks += lastEdgeListSize;


			CustomArray::Array1D<GraphEdge> *combinedArray;
			combinedArray = new CustomArray::Array1D<GraphEdge>(total_edges_in_msf_chunks);
			std::cout << "created combined array" << std::endl;


			Weight done=0;
			for(int i=0; i<numChunks; i++){
				for(int j=0; j<msf_gpu_array[i].Edge_Count; j++){
					GraphEdge g = msf_gpu_array[i].EdgeList->Get(j);
					combinedArray->Set(done++, g);
				}
			}
			std::cout << "---------------------------------------------------";
			std::cout << "last edgelist size " << lastEdgeListSize << std::endl;
			for(int i=0;i<lastEdgeListSize-1;i++){
				GraphEdge g = lastEdgeList->Get(i);
				combinedArray->Set(done++, g);
			}

			SparseGraph* combinedGraph = new SparseGraph(*combinedArray, v, false);

			auto start_final = std::chrono::high_resolution_clock::now();
			Graphs::Algo::MST_Result final_msf = Graphs::Algo::MST_CUDA(combinedGraph);
			auto stop_final = std::chrono::high_resolution_clock::now();
			auto duration_final= std::chrono::duration_cast<std::chrono::microseconds>(stop_final - start_final);
			std::cout << "final chunk execution time: " << duration_final.count() << " microseconds" << std::endl;

			std::cout << "total weight of msf " << final_msf.Total_Cost<< std::endl;
			std::ofstream myfile;
			myfile.open(result_file);
			myfile << final_msf.Total_Cost << "\n";
			myfile << "1st level chunk execution time: " << duration_parallel.count() << " microseconds" << std::endl;
			myfile << "final chunk execution time: " << duration_final.count() << " microseconds" << std::endl;
			myfile << "total execution time " << duration_final.count() + duration_parallel.count() << " microseconds" << std::endl;
			myfile << "number of chunks " << numChunks << std::endl;
			myfile.close();

			

		}


	} // namespace Algo
} // namespace Graphs

#pragma endregion Graph

#endif