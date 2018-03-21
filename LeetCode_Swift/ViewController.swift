//
//  ViewController.swift
//  sss
//
//  Created by Piyush Sharma on 12/13/17.
//  Copyright © 2017 Piyush Sharma. All rights reserved.
//


import UIKit


//a basic vertex data structure

public class Vertex {
    var key: String?
    var neighbors: Array<Edge> = []
    var isVisited: Bool = false

    
    init() {
        self.neighbors = Array<Edge>()
    }
    
}

//a basic edge data structure

public class Edge {
    
    var neighbor: Vertex
    var weight: Int
    
    init() {
        weight = 0
        self.neighbor = Vertex()
    }
    
}

//a default directed graph canvas
public class SwiftGraph {
    
    private var canvas: Array<Vertex>
    public var isDirected: Bool
    
    init() {
        canvas = Array<Vertex>()
        isDirected = true
    }
    
    //create a new vertex
    func addVertex(key: String) -> Vertex {
        
        //set the key
        let childVertex: Vertex = Vertex()
        childVertex.key = key
        
        
        //add the vertex to the graph canvas
        canvas.append(childVertex)
        
        return childVertex
    }
    
    //add an edge to source vertex
    func addEdge(source: Vertex, neighbor: Vertex, weight: Int) {
        
        //create a new edge
        let newEdge = Edge()
        
        //default properties
        newEdge.neighbor = neighbor
        newEdge.weight = weight
        source.neighbors.append(newEdge)
        
        //check for undirected graph
        if (isDirected == false) {
            
            //create a new reversed edge
            let reverseEdge = Edge()
            
            //establish the reversed properties
            reverseEdge.neighbor = source
            reverseEdge.weight = weight
            neighbor.neighbors.append(reverseEdge)
        }
    }
    
    //breadth-first traversal
    func traverseGraphBFS(startingVertex: Vertex) {
        
        //establish a new queue
        var graphQueue: Queue<Vertex> = Queue<Vertex>()
        
        //queue a starting vertex
        graphQueue.enqueue(startingVertex)
        
        while !graphQueue.isEmpty {
            
            //traverse the next queued vertex
            let visitedItem = graphQueue.dequeue()
            
            //add unvisited vertices to the queue
            for nearbyNeighbor in visitedItem!.neighbors {
                if nearbyNeighbor.neighbor.isVisited == false {
                    graphQueue.enqueue(nearbyNeighbor.neighbor)
                }
            }
            visitedItem?.isVisited = true
        } //end while
    } //end function
}


public struct Queue<T> {
    fileprivate var array = [T]()
    
    public var isEmpty: Bool {
        return array.isEmpty
    }
    
    public var count: Int {
        return array.count
    }
    
    public mutating func enqueue(_ element: T) {
        array.append(element)
    }
    
    public mutating func dequeue() -> T? {
        if isEmpty {
            return nil
        } else {
            return array.removeFirst()
        }
    }
    
    public var front: T? {
        return array.first
    }
}





struct HashTable<Key: Hashable,  Value> {
    typealias Element = (key: Key, value: Value)
    typealias Bucket = [Element]
    var buckets = [Bucket]()
    
    private (set) var count = 0
    
    init(capacity: Int) {
        buckets = Array<Bucket>(repeating: [], count: capacity)
    }
    
    func index(for key: Key) -> Int {
        return abs(key.hashValue).quotientAndRemainder(dividingBy: buckets.count).remainder
    }
    
    @discardableResult
    mutating func updateValue(value: Value, for key: Key) -> Value? {
        //first check to see if the value is already inside a bucket. If it is, we update the value at the index
        let keyIndex = index(for: key)
        
        for (i, element) in buckets[keyIndex].enumerated() where element.key == key {
            let oldValue = element.value
            buckets[keyIndex][i].value = value
            return oldValue
        }
        
        /*If execution reaches this point, it means the key doesn't exist inside the hash table.
          we have to add the new key-value pair at the end of the bucket.*/
        buckets[keyIndex].append((key: key, value: value))
        count += 1
        return nil
    }
    
    
    /* This function converts the key into an array index.
       That returns the bucket number, but this bucket may be used by more than one key if there were collisions.
       now we iterate through that index's list of key-value pairs,
       where we compare the key of each element with the key required. */
    
    public func value(for key: Key) -> Value? {
        let keyIndex = self.index(for: key)
        for (i, element) in buckets[keyIndex].enumerated() where element.key == key {
            return buckets[keyIndex][i].value
        }
        return nil
    }
    
    /* we first check to see if the value is in the bucket.
      If it is, remove the key in the chain, decrement the count, and
      return the value. Otherwise, return nil, since we couldn't find the key-value pair to remove.*/
   
    mutating func removeValue(for key : Key) -> Value? {
        //first check to see if the value is already inside a bucket. If it is, we update the value at the index
        let keyIndex = index(for: key)
        
        for (i, element) in buckets[keyIndex].enumerated() where element.key == key {
            buckets[keyIndex].remove(at: i)
            return element.value
        }
        
        count -= 1
        return nil
    }
    
    public subscript(key: Key) -> Value? {
        get {
            return value(for: key)
        }
        set {
            if let newValue = newValue {
                updateValue(value: newValue, for: key)
            }
        }
    }
}



struct RingBuffer<T> {
    var array = [T?]()
    var readIndex = 0
    var writeIndex = 0
    
    init(count: Int) {
        array = Array(repeating: nil, count: count)
    }
    
    var availableSpaceForReading: Int {
        return writeIndex - readIndex
    }
    
    var availbleSpaceForWriting: Int {
        return array.count - availableSpaceForReading
    }
    
    var isFull: Bool {
        return availbleSpaceForWriting == 0
    }
    
    var isEmpty: Bool {
        return availableSpaceForReading ==  0
    }
    
    mutating func write(element: T) -> Bool {
        if !isFull {
            array[writeIndex % array.count] = element
            writeIndex += 1
            return true
        }
        return false
    }
    
    mutating func read() -> T? {
        if !isEmpty {
            let element = array[readIndex % array.count]
            readIndex += 1
            return element
        }
        
        return nil
    }
}



func createArray<T>(value: T, count: Int) -> [T] {
    return Array<T>(repeating: value, count: count)
}



struct ArrayFactory<T> {
    var columns: Int
    var rows: Int
    var array: [T] = []
    
    init(columns: Int, rows: Int, defaultValue: T) {
        self.columns = columns
        self.rows = rows
        array = createArray(value: defaultValue, count: rows*columns)
    }
    
    subscript(column: Int, row: Int) -> T {
        get {
            return array[row*columns + column]
        }
        set {
            array[row*columns + column] = newValue
        }
    }
}



func coinChange(cents: Int) {
    if cents < 0 {
        print("error cenets must be positive")
    } else {
        
        let dollar = cents % 100
        let haldollar = cents % 50
        let quarters = cents / 25
        let dimes = (cents % 25) / 10
        let nickel = ((cents % 25 ) % 10) / 5
        let penny = cents % 5
        
        print(dollar, haldollar)
        print("To make a change of \(cents) we need \(quarters) quarters,  \(dimes) dime, \(nickel) nickel, \(penny) pennies ")
    }
}
//coinChange(cents: 63)
var coins: [Int] = [3,2,5]


//Brute Force Approach
func minChange(coins: [Int], sum: Int) -> Int {
    if sum == 0 {
        return 0
    }
    var result = Int.max
    for coin in coins {
        if coin <= sum {
            result = min(result, minChange(coins: coins, sum: sum - coin) + 1)
        }
    }
    return result
}


//Memoization
class CoinChange {
    var memo:[Int : Int] = [:]
    
    func minNumCoinsForMakingChangeMemo(target: Int, coins: [Int]) -> Int {
        guard memo[target] == nil else {
            return memo[target]!
        }
        if target == 0 {
            return 0 // base case
        }
        var result = Int.max
        for coin in coins {
            if coin <= target {
                result = min(result,  self.minNumCoinsForMakingChangeMemo(target: target - coin, coins: coins) + 1)
                
                print(target-coin)
                memo[target] = result
            }
        }
        
        return result
    }
    
    // Dynamic programming approach for finding minimum number of coins.
    func minNumCoinsForMakingChangeDP(change : Int, coins: [Int]) -> Int {
        var results = [Int](repeating: Int.max, count: change + 1)
        results[0] = 0
        for i in 1...change {
            for coin in coins {
                if i - coin >= 0 {
                    results[i] = min(results[i], results[i - coin] + 1)
                }
            }
        }
        return results[change]
    }
}




//Random Number generator
func random(low: Int, high: Int) -> Int {
    return low + Int(arc4random_uniform(UInt32(high - low + 1)))
}


//Random Selection of numbers from array
func selectionSampling(array: [Int], k: Int) -> [Int] {
    var a = array
    for i in 0..<k {
        let r = random(low: i, high: a.count-1)
        if i != r {
            a.swapAt(i, r)
        }
    }
    return Array(a[0..<k])
}


//The Fisher-Yates / Knuth shuffle
func shuffleArray(array: inout [Int]) {
    for i in (0..<array.count).reversed() {
        let j = random(low: 0, high: i) //here we are limiting our random number generator range on every iteration. (first iteration: (6...0) second iteration: (5...0) so on...)
        if i != j {
            array.swapAt(i, j)
        }
    }
}


func hasPairSum(array: [Int], sum: Int) -> Bool {
    var complements = Set<Int>()
    for value in array {
        if !complements.contains(value) {
            complements.insert(sum-value)
        } else {
            return true
        }
    }
    return false
}



//Heap
struct Heap<T> {
    var elements: [T] = []
    var order: (T,T) -> Bool
    
    init (order: @escaping (T,T) -> Bool) {
        self.order = order
    }
    
    mutating func heapifyAt(index i : Int, heapSize: Int) {
        let left = 2 * i + 1
        let right = 2 * i + 2
        var max = i
        
        if left < heapSize && order(elements[left], elements[max]) {
            max = left
        }
        
        if right < heapSize && order(elements[right], elements[max]) {
            max = right
        }
        
        if i != max {
            elements.swapAt(i, max)
            heapifyAt(index: max, heapSize: heapSize)
        }
    }
    
    mutating func buildHeap(array: [T]) {
        elements = array
        
        for i in stride(from: elements.count/2-1, through: 0, by: -1) {
            heapifyAt(index: i, heapSize: elements.count)
        }
    }
    
    static func heapsort(array: [T], order: @escaping (T,T) -> Bool) -> [T] {
        var heap = Heap<T>(order: order)
        var heapSize = array.count
        
        heap.buildHeap(array: array)
        
        for i in stride(from: heapSize-1, through: 0, by: -1) {
            //we are swapping the top element with the element at last index before reducing heap size and calling of heapify at first index.
            heap.elements.swapAt(0, i)
            heapSize -= 1
            heap.heapifyAt(index: 0, heapSize: heapSize)
        }
        return heap.elements
    }
}


//Swift Quick Sort
func quickSort(array: [Int]) -> [Int] {
    let pivot = array[array.count/2]
    let equal = array.filter { $0 == pivot }
    let less = array.filter { $0 < pivot }
    let greater = array.filter { $0 > pivot }
    return quickSort(array: less) + quickSort(array: greater) + equal
}


//Ducth Flag Partition
func dutchPartition(array: inout [Int], low: Int, high: Int, pivotIndex: Int) -> (Int, Int) {
    let pivot = array[pivotIndex]
  
    var i = low
    var j = low
    var n = high
    
    while j <= n {
        if array[j] < pivot {
            array.swapAt(i, j)
            i += 1
            j += 1
        } else if array[j] == pivot {
            j += 1
        } else {
            array.swapAt(j, n)
            n -= 1
        }
    }
    return (i, n)
}


//Ducth Flag Quick Sort
func quickSortDutch(array: inout [Int], low: Int, high: Int) {
    if low < high {
        let pivotIndex = random(low: low, high: high) //high = array.count -1
        let (p,q) = dutchPartition(array: &array, low: low, high: high, pivotIndex: pivotIndex)
        quickSortDutch(array: &array, low: low, high: p-1)
        quickSortDutch(array: &array, low: q+1, high: high)
    }
}


//Lumoto Partition
func LumotoPartition(array: inout [Int], low: Int, high: Int) -> Int {
    let pivot = array[high]
    var i = low
    
    for j in low..<high {
        if array[j] <= pivot {
            array.swapAt(i, j)
            i += 1
        }
    }
    
    array.swapAt(i, high)
    
    return i
}

//Lumoto Quick Sort
func quickSortLumoto(array: inout [Int], low: Int, high: Int) {
    if low < high {
        let p = LumotoPartition(array: &array, low: low, high: high)
        quickSortLumoto(array: &array, low: low, high: p-1)
        quickSortLumoto(array: &array, low: p+1, high: high)
    }
}

//kth frequent element
//https://github.com/raywenderlich/swift-algorithm-club/tree/master/Kth%20Largest%20Element

func kthEelement(array: [Int], order k: Int) -> Int {
    var a = array
    
    func randomPivot(array: inout [Int], low: Int, high: Int) -> Int {
        let j = random(low: low, high: high)// get random index between low...high
        array.swapAt(high, j) // swap random index element with last element
        return array[high] //return last element
    }
    
    //Apply Lumoto partition on the array around random pivot
    func partition(array: inout [Int], low: Int, high: Int) -> Int {
        let pivot = randomPivot(array: &array, low: low, high: high) //or array[high]
        
        var i = low
        for j in low..<high {
            if array[j] <= pivot {
                array.swapAt(i, j)
                i += 1
            }
        }
        array.swapAt(i, high)
        return i
    }
    
   //Apply binary search after partition of the array
   func randomSelect(array: inout [Int], low: Int, high: Int, k: Int) -> Int {
        let partitionIndex = partition(array: &array, low: low, high: high)
        if low < high {
            if k < partitionIndex {
                return randomSelect(array: &array, low: low, high: partitionIndex-1, k: k)
            } else if k > partitionIndex {
                return randomSelect(array: &array, low: partitionIndex+1, high: high, k: k)
            } else {
                return array[partitionIndex]
            }
         
        } else {
            return array[low]
        }
    }
    //randomSelect(array: &a, low: 0, high: a.count - k - 1, k: k) smallest kth element
    return randomSelect(array: &a, low: 0, high: a.count - 1, k: k) //largest kth element
}








//Find minimum in an array
func minimum<T: Comparable> (array: [T]) -> T? {
    guard var minimum = array.first else {
        return nil
    }
    
    for element in array.dropFirst() {
        minimum = element < minimum ? element : minimum
    }
    
    return minimum
}

//Find maximum in an array
func maximum<T: Comparable>(array: [T]) -> T? {
    guard var maximum = array.first else {
        return nil
    }
    
    for element in array.dropFirst() {
        maximum = element > maximum ? element : maximum
    }
    return maximum
}


//Find maximum/minimum in an array
func maxMin<T: Comparable>(array: [T]) -> (T, T)? {
    // [ 8, 3, 9, 6, 4 ].
    guard var min = array.first else {
        return nil
    }
    
    var max = min //8
    let start = array.count % 2
    
    for i in stride(from: start, to: array.count, by: 2) { //range:  1...4
        let pair = (array[i], array[i+1]) //[ 3, 9 ]
        
        if pair.0 > pair.1 { //compare  pair elements
            
            if pair.0 > max { // compare max with pair's max element
                max = pair.0
            }
            
            if pair.1 < min {
                min = pair.1
            }
        } else {
            
            if pair.1 > max {
                max = pair.1
            }
            
            if pair.0 < min {
                min = pair.0
            }
        }
    }
    return (min, max)
}


//Find occuramce of a num in an array
func countOccurances(array: [Int], key: Int) -> Int {
    
    func rightBondary() -> Int {
        var low = 0
        var high = array.count
        
        let mid = low + (high - low)/2
        while low < high {
            if key < array[mid] {
                high = mid
            } else {
                low = mid + 1
            }
        }
        return low
    }
    
    func leftBondary() -> Int {
        var low = 0
        var high = array.count
        
        let mid = low + (high - low)/2
        while low < high {
            if key > array[mid] {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }
    return rightBondary() - leftBondary()
}





//Merge Sort
func mergeSort(array: [Int]) -> [Int] {
    
    guard array.count > 1 else {
        return array
    }
    let midIndex = array.count/2
    let leftArray = mergeSort(array: Array(array[0..<midIndex]))
    let rightArray = mergeSort(array: Array(array[midIndex..<array.count]))
    return merge(leftPile: leftArray, rightPile: rightArray)
}

//Merging process
func merge(leftPile: [Int], rightPile: [Int]) -> [Int] {
    var orderedArray: [Int] = []
    
    var leftIndex = 0, rightIndex = 0
    
    while leftIndex < leftPile.count && rightIndex < rightPile.count {
        
        if leftPile[leftIndex] < rightPile[rightIndex] {
            orderedArray.append(leftPile[leftIndex])
            leftIndex += 1
        } else if leftPile[leftIndex] > rightPile[rightIndex] {
            orderedArray.append(rightPile[rightIndex])
            rightIndex += 1
        } else {
            orderedArray.append(leftPile[leftIndex])
            leftIndex += 1
            
            orderedArray.append(rightPile[rightIndex])
            rightIndex += 1
        }
        
        while leftIndex < leftPile.count {
            orderedArray.append(leftPile[leftIndex])
            leftIndex += 1
        }
        
        while rightIndex < rightPile.count {
            orderedArray.append(rightPile[rightIndex])
            rightIndex += 1
        }
    }
    return orderedArray
}



//Linear Search
func linearSeacrh<T: Equatable>(array: [T], object: T) -> Int? {
    for (index, obj) in array.enumerated() where obj == object {
        return index
    }
    return nil
}


//Iterative Binary Search
func binarySearch<T: Comparable>(array: [T], key: T) -> Int? {
   var lowerBound = 0
   var upperBound = array.count //i.e 19 elements, last index willl be 18
   let mid = lowerBound + (upperBound-lowerBound)/2
    
    while lowerBound < upperBound {
        if key == array[mid] {
            return mid
        } else if key > array[mid] {
            lowerBound = mid + 1
        } else {
            upperBound = mid
        }
    }
    return nil
}


//recursive
func binarySerch<T: Comparable> (array: [T], key: T, range: Range<Int>) -> Int? {
    if range.lowerBound >= range.upperBound {
        return nil
    }
    
    let midIndex = range.lowerBound + (range.upperBound-range.lowerBound)/2
    if key < array[midIndex] {
        return binarySerch(array: array, key: key, range: 0..<midIndex)
    } else if key > array[midIndex] {
        return binarySerch(array: array, key: key, range: midIndex+1..<range.upperBound)
    } else {
        return midIndex
    }
}






class BinaryTreeNode {
    var value: Int
    var leftChild: BinaryTreeNode?
    var rightChild: BinaryTreeNode?
    
    init (value: Int) {
        self.value = value
    }
}

class BinarySearchTree {
    
    var root: BinaryTreeNode?
    
    /* Function to find LCA of n1 and n2. The function assumes that both
     n1 and n2 are present in BST */
    func findLCA(node: BinaryTreeNode?, n1: Int, n2: Int) -> BinaryTreeNode? {
        if node == nil {
            return nil
        }
        
        //traverse left: n1 and n2 are smaller than root
        if (node!.value > n1 && node!.value > n2) {
            print("left: n1: \(n1) n2: \(n2) node \(node!.value)")
            return findLCA(node: node?.leftChild, n1: n1, n2: n2)
        }
        
        //traverse right: n1 and n2 are greater than root
        if (node!.value < n1 && node!.value < n2) {
            print("right: n1: \(n1) n2: \(n2) node \(node!.value)")
            return findLCA(node: node?.rightChild, n1: n1, n2: n2)
        }
        return node
    }
}
























private extension String {
    subscript(index: Int) -> Character {
        return self[self.index(self.startIndex, offsetBy: index)]
    }
}

private let mapping: [String] = [
    "0",
    "1",
    "abc",
    "def",
    "ghi",
    "jkl",
    "mno",
    "pqrs",
    "tuv",
    "wxyz"
]

struct Medium_017_Letter_Combinations_Of_A_Phone_Number {
    // t = O(3^N), s = O(3^N)
    static func letterCombinations(_ digits: String) -> [String] {
        var combinations: [String] = []
        combinations.append("")
        for i in 0 ..< digits.count {
            let intValue: Int = Int(String(digits[i]))!
            while combinations.first?.count == i {
                //to make combinations of this char store in temp
                let tmp: String = combinations.removeFirst()
                for character in mapping[intValue] {
                    combinations.append("\(tmp)\(character)")
                }
            }
        }
        return combinations
    }
}












struct q14 {
    
    class Solution {
        static var temp = 1
        
        func longestCommonPrefix(_ strs: [String]) -> String {
            var s: String?          //Find the shortest string
            var length = Int.max    //Shortest string's length
            
            for str in strs {
                if str.count < length {
                    length = str.count
                    s = str
                }
            }
            
            if var s = s {
                var endIndex = s.endIndex
                for str in strs {
                    while !s.isEmpty && !str.hasPrefix(s) {
                        endIndex = s.index(before: endIndex)
                        print(endIndex.encodedOffset)
                        s = s.substring(to: endIndex)
                    }
                }
                return s
            } else {
                return ""
            }
        }
    }
    
    static func getSolution() -> Void {
        print(Solution().longestCommonPrefix(["geeksforgeeks", "geeks",
                                              "geek", "geezer"]))
    }
    
    
    static func findDuplicates(array: inout [Int]) {
        var set: Set = Set<Int>()
        for i in 0..<array.count {
            let index = abs(array[i])-1
            
            if array[index] < 0 {
                set.insert(abs(array[i]))
            } else {
                array[index] = -array[index]
            }
        }
    }
    
    static func productSelf(numbersArray: [Int]) {
        var product: [Int] = []
        var temp = 1
        
        /* In this loop, temp variable contains product of
         elements on left side excluding numbersArray[index] */
        for index in stride(from: 0, through: numbersArray.count-1, by: 1) {
            product.append(temp)
            temp  *= numbersArray[index]
        }
        
        /* Initialize temp to 1 for product on right side */
        temp = 1
        
        /* In this loop, temp variable contains product of
         elements on right side excluding numbersArray[index] */
        for index in stride(from: numbersArray.count-1, through: 0, by: -1) {
            product[index] *= temp
            temp *= numbersArray[index]
        }
        
        for index in 0..<numbersArray.count {
            print(product[index])
        }
    }
}






public class ListNode<T: Comparable> {
    var value: T?
    var next: ListNode?
    weak var previous: ListNode?
    
    public init(value: T?) {
        self.value = value
    }
}


public class LinkedList<T: Comparable> {
    
    public typealias Node = ListNode<T>
    
    private var head: Node?
    
    public var isEmpty: Bool {
        return head == nil
    }
    
    public var first: Node? {
        return head
    }
    
    public var last: Node? {
        guard var node = head else {
            return nil
        }
        
        while let next = node.next {
            node = next
        }
        return node
    }
    
    public func append(value: T) {
        let newNode = Node(value: value)
        if let lastNode = last {
            newNode.previous = lastNode
            lastNode.next = newNode
        } else {
            head = newNode
        }
    }
    
    public subscript(index: Int) -> T {
        let node = self.node(atIndex: index)
        return node.value!
    }
    
    public func node(atIndex index: Int) -> Node {
        if index == 0 {
            return head!
        } else {
            var node = head!.next
            for _ in 1..<index {
                node = node?.next
                if node == nil { //(*1)
                    break
                }
            }
            return node!
        }
    }
}


class ViewController: UIViewController {
    
    func combinedIntervals(intervals: [CountableClosedRange<Int>]) -> [CountableClosedRange<Int>] {
        
        var combined = [CountableClosedRange<Int>]()
        var accumulator = (0...0) // empty range
        
        for interval in intervals.sorted(by: { $0.lowerBound  < $1.lowerBound  } ) {
            
            if accumulator == (0...0) {
                accumulator = interval
            }
            
            if accumulator.upperBound >= interval.upperBound {
                // interval is already inside accumulator
            }
                
            else if accumulator.upperBound >= interval.lowerBound  {
                // interval hangs off the back end of accumulator
                accumulator = (accumulator.lowerBound...interval.upperBound)
            }
                
            else if accumulator.upperBound <= interval.lowerBound  {
                // interval does not overlap
                combined.append(accumulator)
                accumulator = interval
            }
        }
        
        if accumulator != (0...0) {
            combined.append(accumulator)
        }
        
        return combined
    }
    
    var shortestString: String? //Find the shortest string
    
    func findMinLength(strArray: [String]) -> Int{
        var length = Int.max
        
        for str in strArray where str.count < length {
            length = str.count
            shortestString = str
        }
        
        return length
    }
    
    func allContainsPrefix(arr: [String], string: String, start: Int, end: Int, length: Int) -> Bool{
        for i in 0...length-1 {
            for j in start..<end {
                let indexI = string.index(string.startIndex, offsetBy: i)
                let indexJ = string.index(string.startIndex, offsetBy: j)
                if string[indexI] != string[indexJ] {
                    return false
                }
            }
        }
        return true
    }
    
    func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        let values = Set(nums1)
        var result = Set<Int>()
        for num in nums2 {
            if values.contains(num) {
                result.insert(num)
            }
        }
        return Array(result)
    }
    
    func mergeTwoLists<T>(_ l1: ListNode<T>?, _ l2: ListNode<T>?) -> ListNode<T>? {
        if l1 == nil { return l2 }
        if l2 == nil { return l1 }
        
        if l1!.value! < l2!.value! {
            l1?.next = mergeTwoLists(l1?.next, l2)
            
            return l1
        } else {
            l2?.next = mergeTwoLists(l1, l2?.next)
            return l2
        }
    }
    
    
    var list: [Int] = []
    var hashMap: [Int:Int] = [:]
    
    func add(element: Int) {
        // If element is already present, then noting to do
        if let _ = hashMap[element] {
            return
        }
        
        //put element at the end of list
        list.append(element)

         //element is added as key and last array index as index.)
        hashMap.updateValue(list.count-1, forKey: element)
    }
    
    func remove(element: Int) {
        /* (1) Check if x is present by doing a hash map lookup.
           (2) If present, then find its index and remove it from hash map.
           (3) Swap the last element with this element in array and remove the last element.
           (4) Update index of last element in hash map. */
        
        //(1) Check if x is present by doing a hash map lookup.
        guard let elementeIndex = hashMap[element] else {
            return
        }
        
        //(2) If present, then find its index and remove it from hash map.
        hashMap.removeValue(forKey: element)
        
        //(3) Swap the last element with this element in array and remove the last element.
        list.swapAt(elementeIndex, list.count-1)
        //Remove last element (This is O(1))
        list.remove(at: list.count-1)

        //(4) Update index of last element in hash map. replace elementeIndex with new index (list.count-1) */
        hashMap.updateValue(list.count-1, forKey: elementeIndex)
    }
    
    func random() -> Int {
        // Find a random index from 0 to list count - 1
        let random = arc4random_uniform(UInt32(list.count))
        return list[Int(random)]
    }
    
    // Returns index of element if element is present, otherwise nil
    func search(element: Int) -> Int? {
        guard let index = hashMap[element] else {
            return nil
        }
        return index
    }
    
  
    func fisherRandom(array: [Int]) -> [Int] {
        var a = array
        for i in array {
            let random = Int(arc4random_uniform(UInt32(a.count - 1)))
            if i != random {
                a.swapAt(i, random)
            }
        }
        return Array(a[0..<a.count])
    }
    
    var previousRandoms = Set<UInt32>()
    func getUniqueRandom(num: Int) -> UInt32 {
        var random = arc4random_uniform(UInt32(num))
       
        while previousRandoms.contains(random) {
            random = arc4random_uniform(UInt32(num))
        }       
        previousRandoms.insert(random)
        print(random)
        return random
    }
    
    override func viewDidLoad() {
        
        var hashTable = HashTable<String, String>(capacity: 5)
//
        hashTable["firstName"] = "Steve"
        hashTable["lastName"] = "Jobs"
        hashTable["hobbies"] = "Programming Swift"

        print(hashTable)

        if let firstName = hashTable["firstName"] {
            print(firstName)
        }
        
        if let lastName = hashTable["lastName"] {
            print(lastName)
        } else {
            print("lastName key not in hash table")
        }

        let x = hashTable["firstName"]
        hashTable["firstName"] = "Tim"
        
        let y = hashTable["firstName"]
        hashTable["firstName"] = nil
        
        let z = hashTable["firstName"]
        
        print(x, y , z)
//        let columns = 4
//        let rows = 4
//        let randomArr = fisherRandom(array: Array(0..<16))
//
//        var cookies = ArrayFactory(columns: columns, rows: rows, defaultValue: 0)
//        for i: Int in 0..<columns {
//            var slicedArray = Array(randomArr[(i*rows)..<rows*columns])
//            for j: Int in 0..<rows {
//                print(slicedArray)
//                cookies[i, j] = slicedArray[j]
//            }
//        }
//
//        print(cookies.array)
//
        
//        add(element: 8)
//        add(element: 2)
//        add(element: 4)
//        add(element: 6)
//        add(element: 9)
//        add(element: 1)
//        add(element: 0)
//
//        remove(element: 0)
        //print(search(element: 4))

        //print(list)

        super.viewDidLoad()
        //let len = lengthOfLongestSubstring(s: "abcabcbb")
        //print(len)
        //var  nums = [0, 1, 0, 3, 12]
        //moveZeroes(&nums)
        //let top = [1,1,2,2,2,3]
        //let arr = topKFrequent(nums: top, 2)
        //print(arr)
        
        
        //        let l1 = LinkedList<Int>()
        //        l1.append(value: 5)
        //        l1.append(value: 7)
        //        l1.append(value: 9)
        //
        //        let l2 = LinkedList<Int>()
        //        l2.append(value: 4)
        //        l2.append(value: 6)
        //        l2.append(value: 8)
        //
        //        let mylist = mergeTwoLists(l1.first, l2.first)
        //        var current: ListNode? = mylist
        //        while current?.value != nil && current != nil {
        //            print("The item is \(String(describing: current?.value))")
        //            current = current?.next
        //        }
        //
        //
        //let inee = intersection([2,2,3,4,4], [1,9,4,5])
        //print(inee)
        
        //let a = findMinLength(strArray: ["geeksforgeeks", "geeks",
        //"geek", "geezer"])
        //print(a)
        
        //let lcp = q14.getSolution()
        
        //let tt = q14.productSelf(numbersArray: [1,2,3,4])
        //var array = [3,3,4]
        //q14.findDuplicates(array: &array)
        
//        let test1 = [(1...3), (2...6), (8...10), (7...11)]
//
//        let result = combinedIntervals(intervals: test1)
//        print(result)
        
        //let comb = Medium_017_Letter_Combinations_Of_A_Phone_Number.letterCombinations("23")
      //  print(comb)
        
//        var tree = BinarySearchTree()
//        tree.root = BinaryTreeNode(value: 20)
//        tree.root?.leftChild = BinaryTreeNode(value: 8)
//        tree.root?.rightChild = BinaryTreeNode(value: 22)
//        tree.root?.leftChild?.leftChild = BinaryTreeNode(value: 4)
//        tree.root?.leftChild?.rightChild = BinaryTreeNode(value: 12)
//        tree.root?.leftChild?.rightChild?.leftChild = BinaryTreeNode(value: 10)
//        tree.root?.leftChild?.rightChild?.rightChild = BinaryTreeNode(value: 14)
//
//        var n1 = 10, n2 = 14;
//        var t = tree.findLCA(node: tree.root, n1: n1, n2: n2)
//        print("LCA of \(n1) and \(n2) is \(t?.value)")
        
        
//        let a = [9, 7, 5, 2, 8, 6, 4]
//
//        print(kthEelement(array: a, order: 0))
//        print(kthEelement(array: a, order: 1))
//        print(kthEelement(array: a, order: 2))
//        print(kthEelement(array: a, order: 3))
//        print(kthEelement(array: a, order: 4))
//        print(kthEelement(array: a, order: 5))
//        print(kthEelement(array: a, order: 6))

        
        
//        var list = [ 10, 0, 3, 9, 2, 14, 8, 27, 1, 5, 8, -1, 26 ]
//        //dutchPartition(array: &list, low: 0, high: list.count-1, pivotIndex: 10)
//        quickSortDutch(array: &list, low: 0, high: list.count-1)
//        print(list)
        
//        var list = [ 10, 0, 3, 9, 2, 14, 26, 27, 1, 5, 8, -1, 8 ]
//        quickSortLumoto(array: &list, low: 0, high: list.count - 1)
//        print(list)
//        binarySearch(numbers, key: 43, range: 0 ..< numbers.count)
//        let arr = Heap.heapsort(array:[3, 5, 8, 1, 2, 9], order: >) // [1, 2, 3, 5, 8, 9]
//        print(arr)
        
//        var array = [1, 2, 3, 4, 5, 6, 7]
//        shuffleArray(array: &array)
//        //let bool = hasPairSum(array: array, sum: 8)
//        print(array)
//
//        let resut = minChange(coins: [1, 3, 4], sum: 5)
//        print(resut)
        
        let change = CoinChange()
        let value  = change.minNumCoinsForMakingChangeDP(change: 63, coins: [25,10,5,1])
        print(value)
//
    }
    
    class WrappedCache<Key, Value> where Key: AnyObject, Value: AnyObject {
        
        let cache = NSCache<Key, Value>()
        
        subscript(key: Key) -> Value? {
            get {
                return cache.object(forKey: key)
            }
            set {
                if let newValue = newValue {
                    cache.setObject(newValue, forKey: key)
                }
                else {
                    cache.removeObject(forKey: key)
                }
            }
        }
    }
    
    
    func topKFrequent(nums: [Int], _ k: Int) -> [Int] {
        var map = [Int: Int]()
        
        for num in nums {
            guard let times = map[num] else {
                map[num] = 1
                continue
            }
            map[num] = times + 1
        }
        
        var keys = Array(map.keys)
        keys.sort() {
            let value1: Int = map[$0]!
            let value2: Int = map[$1]!
            return value1 > value2
        }
        
        return Array(keys[0..<k])
    }
    
    func moveZeroes(_ nums: inout [Int]) {
        var idx = 0
        
        for (i, num) in nums.enumerated() {
            if num != 0 {
                nums[idx] = num
                idx += 1
            }
        }
        
        while idx < nums.count {
            nums[idx] = 0
            idx += 1
        }
        
    }
    
    func lengthOfLongestSubstring(s: String) -> Int {
        guard s.characters.count != 0 else {
            return 0
        }
        
        var set = Set<Character>()
        var maxLen = 0
        var startIndex = 0
        var chars = [Character](s.characters)
        
        for i in 0..<chars.count {
            var current = chars[i]
            
            if set.contains(current) {
                maxLen = max(maxLen, i - startIndex)
                while chars[startIndex] != current {
                    set.remove(chars[startIndex])
                    startIndex += 1
                }
                startIndex += 1
            } else {
                set.insert(current)
            }
        }
        
        maxLen = max(maxLen, chars.count - startIndex)
        
        return maxLen
    }
}
