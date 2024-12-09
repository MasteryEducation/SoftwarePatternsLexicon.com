---
canonical: "https://softwarepatternslexicon.com/patterns-js/15/14"

title: "Web Components and Custom Elements: Building Reusable UI Elements with JavaScript"
description: "Explore Web Components and Custom Elements in JavaScript, including key features like Shadow DOM and HTML Templates, browser support, and libraries like LitElement and Stencil for creating reusable UI components."
linkTitle: "15.14 Web Components and Custom Elements"
tags:
- "Web Components"
- "Custom Elements"
- "JavaScript"
- "Shadow DOM"
- "HTML Templates"
- "LitElement"
- "Stencil"
- "UI Development"
date: 2024-11-25
type: docs
nav_weight: 164000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 15.14 Web Components and Custom Elements

### Introduction

In modern web development, creating reusable and encapsulated UI components is essential for building scalable and maintainable applications. Web Components provide a set of standards that enable developers to create custom, reusable HTML elements with encapsulated functionality. This section will explore the core concepts of Web Components, including Custom Elements, Shadow DOM, and HTML Templates. We will also delve into creating custom elements, discuss browser support and polyfills, and highlight libraries like LitElement and Stencil that simplify working with Web Components. Finally, we'll discuss the benefits of using Web Components and their interoperability with popular frameworks like React and Angular.

### What Are Web Components?

Web Components are a suite of different technologies that allow you to create reusable and encapsulated HTML elements. They consist of three main technologies:

1. **Custom Elements**: Define new HTML elements.
2. **Shadow DOM**: Encapsulate the internal structure of a component.
3. **HTML Templates**: Define reusable HTML fragments.

These technologies work together to enable developers to create complex, reusable UI components that can be used across different projects and frameworks.

### Key Features of Web Components

#### Custom Elements

Custom Elements allow developers to define new HTML tags that encapsulate functionality and behavior. They are created using the `CustomElementRegistry` API, which provides methods to define and register new elements.

##### Creating a Custom Element

To create a custom element, you need to define a class that extends `HTMLElement` and then register it with a unique tag name.

```javascript
// Define a class for the custom element
class MyCustomElement extends HTMLElement {
  constructor() {
    super();
    // Element functionality goes here
  }

  connectedCallback() {
    this.innerHTML = "<p>Hello, I am a custom element!</p>";
  }
}

// Register the custom element
customElements.define('my-custom-element', MyCustomElement);
```

In this example, we define a custom element `<my-custom-element>` that displays a simple message when added to the DOM.

#### Shadow DOM

The Shadow DOM provides encapsulation for the internal structure of a component, preventing styles and scripts from leaking in or out. This ensures that the component's internal structure is isolated from the rest of the document.

##### Using Shadow DOM

To use the Shadow DOM, you attach a shadow root to the custom element and add content to it.

```javascript
class ShadowElement extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    shadow.innerHTML = `
      <style>
        p {
          color: blue;
        }
      </style>
      <p>This is inside the shadow DOM!</p>
    `;
  }
}

customElements.define('shadow-element', ShadowElement);
```

In this example, the `<shadow-element>` uses the Shadow DOM to encapsulate its styles and content, ensuring that the styles do not affect the rest of the document.

#### HTML Templates

HTML Templates allow you to define reusable HTML fragments that can be cloned and inserted into the document. Templates are defined using the `<template>` tag and are not rendered until they are explicitly added to the DOM.

##### Using HTML Templates

```html
<template id="my-template">
  <style>
    .template-content {
      color: red;
    }
  </style>
  <div class="template-content">This is a template!</div>
</template>

<script>
  const template = document.getElementById('my-template');
  const clone = document.importNode(template.content, true);
  document.body.appendChild(clone);
</script>
```

In this example, we define a template with some styled content and then clone it to add it to the document.

### Browser Support and Polyfills

Web Components are supported in most modern browsers, but there are still some compatibility issues with older versions. To ensure compatibility across all browsers, you can use polyfills like the [WebComponents.js](https://github.com/webcomponents/polyfills/tree/master/packages/webcomponentsjs) library.

### Libraries for Web Components

Several libraries simplify the process of creating Web Components, providing additional features and syntactic sugar.

#### LitElement

[LitElement](https://lit.dev/) is a lightweight library that builds on the Web Components standards, making it easier to create fast, lightweight components. It provides a simple base class for creating components with reactive properties and declarative templates.

```javascript
import { LitElement, html, css } from 'lit';

class MyLitElement extends LitElement {
  static styles = css`
    p {
      color: green;
    }
  `;

  render() {
    return html`<p>Hello from LitElement!</p>`;
  }
}

customElements.define('my-lit-element', MyLitElement);
```

#### Stencil

[Stencil](https://stenciljs.com/) is a compiler that generates Web Components and provides features like JSX support, TypeScript integration, and server-side rendering. It is particularly useful for building design systems and component libraries.

### Benefits of Using Web Components

- **Reusability**: Web Components can be reused across different projects and frameworks.
- **Encapsulation**: The Shadow DOM provides encapsulation, preventing styles and scripts from leaking.
- **Interoperability**: Web Components can be used with any JavaScript framework, including React, Angular, and Vue.js.
- **Standardization**: Web Components are based on web standards, ensuring long-term compatibility.

### Interoperability with Frameworks

Web Components can be used with popular frameworks like React and Angular, allowing you to integrate them into existing projects.

#### Using Web Components with React

React can render Web Components just like any other HTML element. However, you may need to handle properties and events differently.

```javascript
import React from 'react';

class App extends React.Component {
  render() {
    return <my-custom-element />;
  }
}

export default App;
```

#### Using Web Components with Angular

Angular provides support for Web Components through its `CUSTOM_ELEMENTS_SCHEMA`.

```typescript
import { NgModule, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';

@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  bootstrap: [AppComponent]
})
export class AppModule {}
```

### Conclusion

Web Components provide a powerful way to create reusable and encapsulated UI elements. By leveraging Custom Elements, Shadow DOM, and HTML Templates, developers can build components that are both flexible and maintainable. Libraries like LitElement and Stencil further simplify the process, making it easier to create complex components. With broad browser support and interoperability with popular frameworks, Web Components are an essential tool for modern web development.

### Try It Yourself

Experiment with creating your own custom elements and using the Shadow DOM. Try integrating Web Components into a React or Angular project to see how they can enhance your application's architecture.

### Knowledge Check

## Web Components and Custom Elements Quiz

{{< quizdown >}}

### What are the three main technologies that make up Web Components?

- [x] Custom Elements, Shadow DOM, HTML Templates
- [ ] Custom Elements, Shadow DOM, CSS Variables
- [ ] Shadow DOM, HTML Templates, CSS Grid
- [ ] Custom Elements, HTML Templates, CSS Flexbox

> **Explanation:** Web Components consist of Custom Elements, Shadow DOM, and HTML Templates.

### How do you define a new custom element in JavaScript?

- [x] By extending HTMLElement and using customElements.define()
- [ ] By using the `createElement` method
- [ ] By defining a new HTML tag in the HTML file
- [ ] By using the `document.registerElement` method

> **Explanation:** Custom elements are defined by extending `HTMLElement` and registering them with `customElements.define()`.

### What is the purpose of the Shadow DOM?

- [x] To encapsulate the internal structure of a component
- [ ] To provide a global style for all components
- [ ] To create a new HTML element
- [ ] To define reusable HTML fragments

> **Explanation:** The Shadow DOM encapsulates the internal structure of a component, preventing styles and scripts from leaking.

### Which library is known for simplifying the creation of Web Components with reactive properties?

- [x] LitElement
- [ ] React
- [ ] Angular
- [ ] Vue.js

> **Explanation:** LitElement is a library that simplifies the creation of Web Components with reactive properties.

### How can you use Web Components in a React application?

- [x] By rendering them like any other HTML element
- [ ] By converting them into React components
- [ ] By using a special React plugin
- [ ] By wrapping them in a React component

> **Explanation:** Web Components can be rendered in React applications like any other HTML element.

### What is the role of HTML Templates in Web Components?

- [x] To define reusable HTML fragments
- [ ] To encapsulate styles
- [ ] To create new HTML elements
- [ ] To provide global styles

> **Explanation:** HTML Templates define reusable HTML fragments that can be cloned and inserted into the document.

### Which of the following is a benefit of using Web Components?

- [x] Reusability across different projects and frameworks
- [ ] Limited browser support
- [ ] Incompatibility with JavaScript frameworks
- [ ] Increased complexity in styling

> **Explanation:** Web Components offer reusability across different projects and frameworks.

### What is the purpose of using polyfills with Web Components?

- [x] To ensure compatibility with older browsers
- [ ] To enhance performance in modern browsers
- [ ] To add new features to Web Components
- [ ] To simplify the creation of Web Components

> **Explanation:** Polyfills ensure compatibility with older browsers that do not fully support Web Components.

### Which schema is used in Angular to support Web Components?

- [x] CUSTOM_ELEMENTS_SCHEMA
- [ ] NO_ERRORS_SCHEMA
- [ ] DEFAULT_SCHEMA
- [ ] COMPONENT_SCHEMA

> **Explanation:** Angular uses the `CUSTOM_ELEMENTS_SCHEMA` to support Web Components.

### True or False: Web Components can only be used with vanilla JavaScript.

- [ ] True
- [x] False

> **Explanation:** Web Components can be used with any JavaScript framework, including React, Angular, and Vue.js.

{{< /quizdown >}}

Remember, this is just the beginning. As you progress, you'll build more complex and interactive web pages. Keep experimenting, stay curious, and enjoy the journey!

---
