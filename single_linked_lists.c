#include <stdio.h>
#include <stdlib.h>

// Node structure
struct Node {
    int data;
    struct Node* next;
};

// Function to add a new node at the end of the linked list
void addNode(struct Node** head, int data) {
    // Create a new node
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;

    // If the list is empty, make the new node the head
    if (*head == NULL) {
        *head = newNode;
        return;
    }

    // Traverse to the last node and append the new node
    struct Node* current = *head;
    while (current->next != NULL) {
        current = current->next;
    }
    current->next = newNode;
}

// Function to delete a particular node by location
void deleteNodeByLocation(struct Node** head, int location) {
    if (*head == NULL) {
        printf("List is empty.\n");
        return;
    }

    struct Node* temp = *head;

    // If the head node needs to be deleted
    if (location == 0) {
        *head = temp->next;
        free(temp);
        return;
    }

    // Find the previous node of the node to be deleted
    for (int i = 0; temp != NULL && i < location - 1; i++) {
        temp = temp->next;
    }

    // If the location is greater than the number of nodes
    if (temp == NULL || temp->next == NULL) {
        printf("Invalid location.\n");
        return;
    }

    // Node temp->next is the node to be deleted
    struct Node* nextNode = temp->next->next;
    free(temp->next);
    temp->next = nextNode;
}

// Function to delete all nodes containing a particular data
void deleteNodesByData(struct Node** head, int data) {
    struct Node* current = *head;
    struct Node* temp = NULL;

    // Delete nodes at the beginning with the particular data
    while (current != NULL && current->data == data) {
        temp = current;
        *head = current->next;
        current = current->next;
        free(temp);
    }

    // Delete nodes in the middle or at the end with the particular data
    while (current != NULL) {
        while (current != NULL && current->data != data) {
            temp = current;
            current = current->next;
        }
        if (current == NULL) {
            return;
        }
        temp->next = current->next;
        free(current);
        current = temp->next;
    }
}

// Function to delete the complete linked list
void deleteLinkedList(struct Node** head) {
    struct Node* current = *head;
    struct Node* nextNode = NULL;

    while (current != NULL) {
        nextNode = current->next;
        free(current);
        current = nextNode;
    }

    *head = NULL;
}

// Function to display the linked list
void displayLinkedList(struct Node* head) {
    struct Node* current = head;

    if (current == NULL) {
        printf("List is empty.\n");
        return;
    }

    printf("Linked List: ");
    while (current != NULL) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("\n");
}

// Function to display the inverted linked list
void displayInvertedLinkedList(struct Node* head) {
    if (head == NULL) {
        printf("List is empty.\n");
        return;
    }

    if (head->next == NULL) {
        printf("Inverted Linked List: %d\n", head->data);
        return;
    }

    displayInvertedLinkedList(head->next);
    printf("%d ", head->data);
}

// Function to display the total memory space occupied by the linked list
void displayMemorySpace(struct Node* head) {
    struct Node* current = head;
    int count = 0;

    while (current != NULL) {
        count++;
        current = current->next;
    }

    printf("Total memory space occupied by the linked list: %d bytes\n", count * sizeof(struct Node));
}

// Main function to test the linked list functions
int main() {
    struct Node* head = NULL;

    addNode(&head, 1);
    addNode(&head, 2);
    addNode(&head, 3);
    addNode(&head, 4);
    addNode(&head, 5);
    addNode(&head, 5);
    addNode(&head, 6);
    addNode(&head, 5);

    displayLinkedList(head);
    displayInvertedLinkedList(head);
    displayMemorySpace(head);

    deleteNodeByLocation(&head, 3);
    deleteNodesByData(&head, 5);

    displayLinkedList(head);
    displayInvertedLinkedList(head);
    displayMemorySpace(head);

    deleteLinkedList(&head);

    return 0;
}
