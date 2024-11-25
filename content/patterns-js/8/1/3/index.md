---

linkTitle: "8.1.3 State Management with NgRx"
title: "Mastering State Management with NgRx in Angular"
description: "Explore how to implement state management in Angular applications using NgRx, focusing on store, actions, reducers, and effects for efficient state handling."
categories:
- Angular
- State Management
- NgRx
tags:
- Angular
- NgRx
- State Management
- TypeScript
- Frontend Development
date: 2024-10-25
type: docs
nav_weight: 8130

canonical: "https://softwarepatternslexicon.com/patterns-js/8/1/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 8.1.3 Mastering State Management with NgRx in Angular

State management is a critical aspect of modern web applications, especially as they grow in complexity. Angular, a popular framework for building dynamic web applications, offers NgRx as a powerful solution for managing state. In this section, we'll delve into the concepts, implementation steps, and best practices for using NgRx in Angular applications.

### Understand the Concepts

NgRx is a state management library for Angular applications, inspired by Redux. It provides a way to manage application state in a predictable and consistent manner. The core concepts of NgRx include:

- **Store:** A centralized state container that holds the application's state.
- **Actions:** Plain objects that describe state changes.
- **Reducers:** Pure functions that specify how the application's state changes in response to actions.
- **Effects:** Handle side effects and asynchronous operations outside of the store.

### Implementation Steps

#### Set Up NgRx Store

To get started with NgRx, you need to install the necessary packages and set up the store in your Angular application.

1. **Install NgRx Packages:**

   Use the Angular CLI to install NgRx packages:

   ```bash
   ng add @ngrx/store @ngrx/effects @ngrx/store-devtools
   ```

2. **Import StoreModule:**

   Import `StoreModule` in your `AppModule` to configure the store:

   ```typescript
   import { NgModule } from '@angular/core';
   import { BrowserModule } from '@angular/platform-browser';
   import { StoreModule } from '@ngrx/store';
   import { EffectsModule } from '@ngrx/effects';
   import { StoreDevtoolsModule } from '@ngrx/store-devtools';
   import { environment } from '../environments/environment';

   @NgModule({
     declarations: [AppComponent],
     imports: [
       BrowserModule,
       StoreModule.forRoot({}, {}),
       EffectsModule.forRoot([]),
       !environment.production ? StoreDevtoolsModule.instrument() : []
     ],
     providers: [],
     bootstrap: [AppComponent]
   })
   export class AppModule {}
   ```

#### Define State and Actions

Next, define the state interface and actions that will be used to manage state changes.

1. **Create State Interface:**

   Define the shape of your application's state using TypeScript interfaces:

   ```typescript
   export interface Todo {
     id: number;
     title: string;
     completed: boolean;
   }

   export interface AppState {
     todos: Todo[];
   }
   ```

2. **Define Action Types:**

   Create action types and action creators to describe state changes:

   ```typescript
   import { createAction, props } from '@ngrx/store';
   import { Todo } from './todo.model';

   export const addTodo = createAction(
     '[Todo] Add Todo',
     props<{ todo: Todo }>()
   );

   export const removeTodo = createAction(
     '[Todo] Remove Todo',
     props<{ id: number }>()
   );

   export const loadTodos = createAction('[Todo] Load Todos');
   ```

#### Create Reducers

Reducers are pure functions that take the current state and an action, and return a new state.

1. **Implement Reducers:**

   Create a reducer function to handle actions and update the state:

   ```typescript
   import { createReducer, on } from '@ngrx/store';
   import { addTodo, removeTodo } from './todo.actions';
   import { Todo } from './todo.model';

   export const initialState: Todo[] = [];

   const _todoReducer = createReducer(
     initialState,
     on(addTodo, (state, { todo }) => [...state, todo]),
     on(removeTodo, (state, { id }) => state.filter(todo => todo.id !== id))
   );

   export function todoReducer(state, action) {
     return _todoReducer(state, action);
   }
   ```

#### Implement Effects

Effects handle side effects such as HTTP requests, and dispatch actions based on the results.

1. **Create Effects:**

   Use `EffectsModule` to define effects for handling asynchronous operations:

   ```typescript
   import { Injectable } from '@angular/core';
   import { Actions, createEffect, ofType } from '@ngrx/effects';
   import { of } from 'rxjs';
   import { map, mergeMap, catchError } from 'rxjs/operators';
   import { TodoService } from './todo.service';
   import { loadTodos, addTodo } from './todo.actions';

   @Injectable()
   export class TodoEffects {
     loadTodos$ = createEffect(() =>
       this.actions$.pipe(
         ofType(loadTodos),
         mergeMap(() =>
           this.todoService.getAll().pipe(
             map(todos => ({ type: '[Todo API] Todos Loaded Success', todos })),
             catchError(() => of({ type: '[Todo API] Todos Loaded Error' }))
           )
         )
       )
     );

     constructor(private actions$: Actions, private todoService: TodoService) {}
   }
   ```

#### Select State

Selectors are used to access specific slices of the state in components.

1. **Create Selectors:**

   Define selectors to retrieve data from the store:

   ```typescript
   import { createSelector, createFeatureSelector } from '@ngrx/store';
   import { AppState, Todo } from './todo.model';

   export const selectTodos = createFeatureSelector<AppState, Todo[]>('todos');

   export const selectCompletedTodos = createSelector(
     selectTodos,
     (todos: Todo[]) => todos.filter(todo => todo.completed)
   );
   ```

2. **Use Selectors in Components:**

   Access state in your components using the defined selectors:

   ```typescript
   import { Component } from '@angular/core';
   import { Store, select } from '@ngrx/store';
   import { Observable } from 'rxjs';
   import { Todo } from './todo.model';
   import { selectTodos } from './todo.selectors';

   @Component({
     selector: 'app-todo-list',
     template: `
       <ul>
         <li *ngFor="let todo of todos$ | async">{{ todo.title }}</li>
       </ul>
     `
   })
   export class TodoListComponent {
     todos$: Observable<Todo[]>;

     constructor(private store: Store) {
       this.todos$ = this.store.pipe(select(selectTodos));
     }
   }
   ```

### Use Cases

NgRx is particularly useful in scenarios where you need to manage complex application state, such as:

- Applications with multiple components that need to share state.
- Applications that require undo/redo functionality.
- Applications that need to persist state across sessions.

### Practice: Develop a Todo App

To solidify your understanding of NgRx, try developing a simple todo application that manages state using NgRx. Implement features such as adding, removing, and filtering todos, and handle asynchronous operations like fetching todos from an API.

### Considerations

- **StoreDevtools:** Use `StoreDevtools` to debug state changes and track actions. This tool provides a time-traveling debugger that helps you understand how state changes over time.

  ```typescript
  import { StoreDevtoolsModule } from '@ngrx/store-devtools';
  import { environment } from '../environments/environment';

  @NgModule({
    imports: [
      StoreDevtoolsModule.instrument({
        maxAge: 25, // Retains last 25 states
        logOnly: environment.production, // Restrict extension to log-only mode
      }),
    ],
  })
  ```

- **Performance:** Be mindful of performance implications when using NgRx, especially in large applications. Use selectors efficiently to minimize unnecessary re-renders.

### Conclusion

NgRx provides a robust framework for managing state in Angular applications. By following the concepts and implementation steps outlined in this article, you can build scalable and maintainable applications with ease. Remember to leverage tools like `StoreDevtools` for debugging and always consider performance implications when designing your state management strategy.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of NgRx in Angular applications?

- [x] To manage application state in a predictable manner
- [ ] To handle HTTP requests
- [ ] To style components
- [ ] To manage routing

> **Explanation:** NgRx is primarily used for state management, providing a predictable way to manage application state.

### Which of the following is NOT a core concept of NgRx?

- [ ] Store
- [ ] Actions
- [ ] Reducers
- [x] Components

> **Explanation:** Components are part of Angular, not a core concept of NgRx. NgRx focuses on store, actions, reducers, and effects.

### What is the role of reducers in NgRx?

- [x] To specify how the application's state changes in response to actions
- [ ] To handle side effects
- [ ] To dispatch actions
- [ ] To select state slices

> **Explanation:** Reducers are pure functions that specify how the application's state changes in response to actions.

### How do effects in NgRx handle asynchronous operations?

- [x] By using the `EffectsModule` to manage side effects
- [ ] By directly modifying the store
- [ ] By creating new components
- [ ] By using Angular services

> **Explanation:** Effects handle side effects and asynchronous operations using the `EffectsModule`.

### What is the purpose of selectors in NgRx?

- [x] To access specific slices of the state in components
- [ ] To dispatch actions
- [ ] To handle HTTP requests
- [ ] To create new reducers

> **Explanation:** Selectors are used to access specific slices of the state in components.

### Which tool can be used for debugging state changes in NgRx?

- [x] StoreDevtools
- [ ] Angular CLI
- [ ] TypeScript
- [ ] NgRx Router

> **Explanation:** StoreDevtools is a tool used for debugging state changes and tracking actions in NgRx.

### What is the benefit of using NgRx in applications with multiple components?

- [x] It allows components to share state efficiently
- [ ] It simplifies component styling
- [ ] It automatically generates components
- [ ] It manages component lifecycle

> **Explanation:** NgRx allows components to share state efficiently, which is beneficial in applications with multiple components.

### In which file is the `StoreModule` typically imported in an Angular application?

- [x] AppModule
- [ ] Main.ts
- [ ] Index.html
- [ ] Angular.json

> **Explanation:** The `StoreModule` is typically imported in the `AppModule` to configure the store.

### What is a common use case for NgRx?

- [x] Managing complex application state
- [ ] Styling components
- [ ] Handling component lifecycle
- [ ] Managing routing

> **Explanation:** NgRx is commonly used for managing complex application state.

### True or False: NgRx can be used to persist state across sessions.

- [x] True
- [ ] False

> **Explanation:** NgRx can be used to persist state across sessions, making it useful for applications that require state persistence.

{{< /quizdown >}}
