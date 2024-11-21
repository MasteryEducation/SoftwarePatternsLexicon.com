---
linkTitle: "Distributed Consensus"
title: "Distributed Consensus: Ensuring Consistency Across Distributed Systems"
description: "Consensus protocols enable multiple nodes in a distributed system to agree on a common state even in the presence of faults."
categories:
- Robust and Reliable Architectures
tags:
- Distributed Systems
- Consensus Protocols
- Fault Tolerance
- Robustness
- Reliability
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/robust-and-reliable-architectures/fault-tolerance/distributed-consensus"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


In distributed systems, achieving a consensus is crucial to ensure the consistency and reliability of the system's state across multiple nodes. **Distributed Consensus** is a design pattern that enables several agents or nodes to come to an agreement on some data value needed for operation, despite the presence of some nodes failing or behaving incorrectly. This concept is particularly vital in fault-tolerant systems and plays a critical role in areas such as data replication, distributed databases, and blockchain technologies.

## Importance of Distributed Consensus

In any distributed system, ensuring data consistency and agreeing on the sequence of operations are fundamental challenges. Without a robust mechanism, inconsistencies can lead to data corruption, service outages, and erroneous computations. Distributed consensus algorithms are designed to address these challenges by providing the following guarantees:
- **Consistency**: All non-faulty nodes will eventually agree on the same value.
- **Fault Tolerance**: The system can tolerate a certain number of node failures without compromising consensus.
- **Liveness**: The system eventually reaches a decision, ensuring continued operation.

## Consensus Protocols

### 1. Paxos

Paxos is one of the most renowned consensus algorithms, characterized by its strong theoretical foundation. It operates through a series of phases involving proposers, acceptors, and learners:
- **Prepare Phase**: A proposer selects a proposal number and sends a prepare request to the majority of acceptors.
- **Promise Phase**: Acceptors respond to the request if they haven't already promised another proposal with a higher number.
- **Accept Phase**: Upon acquiring promises, the proposer sends an accept request with the proposal.
- **Learn Phase**: If the majority of acceptors accept the proposal, it is committed and learned by everyone.

### 2. Raft

Raft is a consensus algorithm designed for understandability and practical implementation. It segments the consensus process into three distinct sub-problems: leader election, log replication, and safety.
- **Leader Election**: One node is elected as the leader to manage all client interactions and log replication.
- **Log Replication**: The leader appends new entries to its log and replicates them to followers.
- **Commitment**: When the majority of followers acknowledge the log entries, they are considered committed.
  
### 3. Byzantine Fault Tolerance (BFT)

BFT algorithms are designed to withstand arbitrary faults, including Byzantine faults where nodes may act maliciously. Practical Byzantine Fault Tolerance (PBFT) is one such algorithm that ensures consensus even under the most adversarial conditions.
- **Pre-prepare Phase**: The primary node proposes a value.
- **Prepare Phase**: Nodes prepare and broadcast their acceptance of the proposal.
- **Commit Phase**: Nodes commit to the proposal after receiving the majority's approval.

## Examples

### Python Example with Raft

```python
import threading
import time

class RaftNode:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers
        self.state = 'follower'  # states: follower, candidate, leader
        self.current_term = 0
        self.voted_for = None
        self.log = []

    def start_election(self):
        self.state = 'candidate'
        self.current_term += 1
        self.voted_for = self.node_id
        # Send RequestVote RPCs to all other nodes
        votes = 1
        for peer in self.peers:
            if self.send_request_vote(peer):
                votes += 1
        if votes > len(self.peers) // 2:
            self.state = 'leader'
            self.heartbeat()

    def send_request_vote(self, peer):
        # Simulate sending a RequestVote RPC
        return True  # Assuming the vote is granted for simplicity

    def heartbeat(self):
        while self.state == 'leader':
            # Leader sends heartbeats to maintain authority
            print(f'{self.node_id} sending heartbeat')
            time.sleep(1)

if __name__ == "__main__":
    node = RaftNode(node_id=1, peers=[2, 3])
    election_thread = threading.Thread(target=node.start_election)
    election_thread.start()
```

### Go Example with Paxos

```go
package main

import (
	"fmt"
	"sync"
)

type Node struct {
	id        int
	acceptors []*Node
	mu        sync.Mutex
	promiseID int
	accepted  int
}

func (n *Node) prepare(proposalID int) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	if proposalID > n.promiseID {
		n.promiseID = proposalID
		return true
	}
	return false
}

func main() {
	n1 := &Node{id: 1}
	n2 := &Node{id: 2}
	n3 := &Node{id: 3}
	n1.acceptors = []*Node{n2, n3}
	n2.acceptors = []*Node{n1, n3}
	n3.acceptors = []*Node{n1, n2}

	proposalID := 10
	for _, acceptor := range n1.acceptors {
		if acceptor.prepare(proposalID) {
			fmt.Printf("Node %d accepted the proposal %d\n", acceptor.id, proposalID)
		} else {
			fmt.Printf("Node %d rejected the proposal %d\n", acceptor.id, proposalID)
		}
	}
}
```

## Related Design Patterns

### **Leader Election**
Leader Election is often a prerequisite in consensus algorithms to determine a single node (the leader) responsible for coordinating tasks, ensuring there's no chaos in operations.

### **Quorum**
Quorum involves ensuring a decision is made only when a majority (or another specific subset) of nodes agree, balancing the need for availability and consistency.

### **Replication**
Replication copies the state or data from a primary source to multiple nodes, which can then agree on the latest state using consensus algorithms.

## Additional Resources

1. Lamport, Leslie. "The part-time parliament." ACM Transactions on Computer Systems (1998)
2. Ongaro, Diego, and John Ousterhout. "In search of an understandable consensus algorithm." USENIX Annual Technical Conference (2014)
3. Castro, Miguel, and Barbara Liskov. "Practical Byzantine Fault Tolerance." OSDI (1999)
4. **[Raft Consensus Algorithm](https://raft.github.io/)**
5. **[Paxos Made Simple](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf)**

## Summary

Distributed consensus protocols are essential for maintaining consistency and reliability in distributed systems, especially in environments prone to faults. Protocols such as Paxos, Raft, and BFT provide frameworks to achieve consensus with varying degrees of fault tolerance and complexity. Understanding and implementing these protocols are fundamental to building robust, fault-tolerant architectures, which are critical in modern distributed applications like databases, data storage systems, and blockchain technologies.
