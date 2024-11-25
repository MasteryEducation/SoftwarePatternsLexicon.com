---
canonical: "https://softwarepatternslexicon.com/patterns-ruby/21/1"
title: "GUI Development with Ruby: Tk, GTK, and Shoes"
description: "Explore GUI development in Ruby using Tk, GTK, and Shoes. Learn to create interactive desktop applications with comprehensive examples and setup guides."
linkTitle: "21.1 GUI Development with Ruby (Tk, GTK, Shoes)"
categories:
- Ruby Development
- GUI Programming
- Desktop Applications
tags:
- Ruby
- GUI
- Tk
- GTK
- Shoes
- Desktop Applications
date: 2024-11-23
type: docs
nav_weight: 211000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 21.1 GUI Development with Ruby (Tk, GTK, Shoes)

Graphical User Interfaces (GUIs) play a crucial role in making applications accessible and user-friendly. In Ruby, several libraries allow developers to create rich, interactive desktop applications. This section introduces three popular GUI libraries: Tk, GTK, and Shoes. We will explore their features, installation, and how to create basic applications with each.

### Importance of GUI Development

GUI development is essential for creating applications that are intuitive and easy to use. A well-designed GUI can significantly enhance user experience by providing visual feedback, simplifying complex tasks, and making applications more accessible to non-technical users. In desktop applications, GUIs are the primary means through which users interact with software, making their design and functionality critical to the application's success.

### Tk: The Default GUI Toolkit

Tk is the default GUI toolkit included with Ruby. It is a mature and stable library that provides a wide range of widgets and is suitable for creating simple to moderately complex applications.

#### Installation and Setup

To use Tk with Ruby, you need to ensure that the Tk library is installed on your system. On most systems, this can be done by installing the `tk` package through your package manager. For example, on Ubuntu, you can use:

```bash
sudo apt-get install tk
```

Once Tk is installed, you can verify its availability in Ruby by requiring it in your script:

```ruby
require 'tk'
```

#### Creating a Basic Window with Tk

Let's create a simple window using Tk to understand its basic usage:

```ruby
require 'tk'

# Create the main window
root = TkRoot.new { title "Hello Tk" }

# Create a label widget
TkLabel.new(root) do
  text 'Welcome to Tk!'
  pack { padx 15; pady 15; side 'left' }
end

# Start the Tk main event loop
Tk.mainloop
```

**Explanation:**

- **TkRoot**: This creates the main application window.
- **TkLabel**: A widget to display text.
- **pack**: A geometry manager to arrange widgets.

#### Pros and Cons of Tk

**Pros:**

- **Cross-Platform**: Works on Windows, macOS, and Linux.
- **Stable and Mature**: Well-tested and reliable.
- **Included with Ruby**: No additional gems required.

**Cons:**

- **Basic Look and Feel**: May not match modern UI aesthetics.
- **Limited Widgets**: Compared to more modern toolkits.

### GTK: A Powerful GUI Toolkit

GTK, accessed via the Ruby-GNOME2 project, is a powerful and flexible GUI toolkit. It provides a rich set of widgets and is suitable for creating complex applications.

#### Installation and Setup

To use GTK with Ruby, you need to install the `gtk3` gem. Ensure you have the necessary GTK libraries installed on your system. On Ubuntu, you can install them with:

```bash
sudo apt-get install libgtk-3-dev
```

Then, install the Ruby bindings:

```bash
gem install gtk3
```

#### Creating a Basic Window with GTK

Here's how to create a simple window using GTK:

```ruby
require 'gtk3'

# Initialize GTK
Gtk.init

# Create a new window
window = Gtk::Window.new
window.set_title("Hello GTK")
window.set_default_size(300, 200)

# Create a label
label = Gtk::Label.new("Welcome to GTK!")
window.add(label)

# Show the window
window.signal_connect("destroy") { Gtk.main_quit }
window.show_all

# Start the GTK main event loop
Gtk.main
```

**Explanation:**

- **Gtk::Window**: Represents the main application window.
- **Gtk::Label**: A widget to display text.
- **signal_connect**: Connects signals to event handlers.

#### Pros and Cons of GTK

**Pros:**

- **Rich Set of Widgets**: Suitable for complex applications.
- **Modern Look**: Provides a contemporary UI.
- **Active Community**: Regular updates and support.

**Cons:**

- **Complexity**: More complex than Tk, with a steeper learning curve.
- **Platform Dependencies**: Requires GTK libraries to be installed.

### Shoes: A Simple and Fun Toolkit

Shoes is a simple and easy-to-use toolkit designed for beginners. It is ideal for creating small applications and prototypes quickly.

#### Installation and Setup

Shoes can be installed by downloading the appropriate version for your operating system from the [Shoes website](http://shoesrb.com/). Follow the installation instructions provided on the site.

#### Creating a Basic Window with Shoes

Here's a simple example of creating a window with Shoes:

```ruby
Shoes.app(title: "Hello Shoes", width: 300, height: 200) do
  para "Welcome to Shoes!"
end
```

**Explanation:**

- **Shoes.app**: Defines the main application window.
- **para**: A widget to display text.

#### Pros and Cons of Shoes

**Pros:**

- **Easy to Learn**: Simple syntax and quick setup.
- **Great for Prototyping**: Rapid development of small applications.

**Cons:**

- **Limited Features**: Not suitable for complex applications.
- **Smaller Community**: Less active development compared to GTK.

### Choosing the Right Toolkit

When deciding which GUI library to use, consider the following:

- **Project Complexity**: For simple applications, Tk or Shoes may suffice. For more complex applications, GTK is more appropriate.
- **Platform Requirements**: Ensure the library supports your target platforms.
- **Development Speed**: Shoes offers rapid development, while GTK provides more control and features.

### Experiment: Build a Simple GUI Application

Try building a simple application using one of these libraries. For example, create a basic calculator or a to-do list application. Experiment with different widgets and layouts to understand the capabilities of each toolkit.

### Further Reading and Resources

- **Tk**: [Tk Documentation](https://www.ruby-lang.org/en/documentation/installation/#ruby-tk)
- **GTK**: [Ruby-GNOME2](https://ruby-gnome2.osdn.jp/)
- **Shoes**: [Shoes Website](http://shoesrb.com/)

### Summary

In this section, we've explored the basics of GUI development in Ruby using Tk, GTK, and Shoes. Each library offers unique features and capabilities, making them suitable for different types of projects. By understanding these tools, you can create interactive and user-friendly desktop applications.

## Quiz: GUI Development with Ruby (Tk, GTK, Shoes)

{{< quizdown >}}

### Which GUI toolkit is included by default with Ruby?

- [x] Tk
- [ ] GTK
- [ ] Shoes
- [ ] Qt

> **Explanation:** Tk is the default GUI toolkit included with Ruby.

### What is a key advantage of using GTK for GUI development?

- [x] Rich set of widgets
- [ ] Simple syntax
- [ ] No dependencies
- [ ] Limited features

> **Explanation:** GTK provides a rich set of widgets, making it suitable for complex applications.

### Which library is best suited for beginners and rapid prototyping?

- [ ] Tk
- [ ] GTK
- [x] Shoes
- [ ] Qt

> **Explanation:** Shoes is designed for beginners and allows for rapid prototyping.

### What command is used to install the GTK libraries on Ubuntu?

- [x] `sudo apt-get install libgtk-3-dev`
- [ ] `gem install gtk3`
- [ ] `sudo apt-get install tk`
- [ ] `gem install shoes`

> **Explanation:** The command `sudo apt-get install libgtk-3-dev` installs the necessary GTK libraries on Ubuntu.

### Which toolkit requires the least setup for a simple application?

- [ ] GTK
- [x] Shoes
- [ ] Tk
- [ ] Qt

> **Explanation:** Shoes requires minimal setup and is ideal for simple applications.

### What is the primary function of the `pack` method in Tk?

- [x] Arrange widgets
- [ ] Create windows
- [ ] Handle events
- [ ] Display text

> **Explanation:** The `pack` method is used to arrange widgets in Tk.

### Which library is accessed via the Ruby-GNOME2 project?

- [ ] Tk
- [x] GTK
- [ ] Shoes
- [ ] Qt

> **Explanation:** GTK is accessed via the Ruby-GNOME2 project.

### What is a disadvantage of using Shoes for GUI development?

- [ ] Easy to learn
- [ ] Great for prototyping
- [x] Limited features
- [ ] Active community

> **Explanation:** Shoes has limited features, making it less suitable for complex applications.

### Which toolkit provides a modern look and feel?

- [ ] Tk
- [x] GTK
- [ ] Shoes
- [ ] Qt

> **Explanation:** GTK provides a modern look and feel.

### True or False: Tk is suitable for creating complex applications.

- [ ] True
- [x] False

> **Explanation:** Tk is more suitable for simple to moderately complex applications.

{{< /quizdown >}}
