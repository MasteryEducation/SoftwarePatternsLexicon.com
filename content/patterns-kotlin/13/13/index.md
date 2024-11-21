---
canonical: "https://softwarepatternslexicon.com/patterns-kotlin/13/13"
title: "Handling Configuration Changes in Android Development"
description: "Explore strategies and best practices for managing configuration changes in Android apps using Kotlin, including state management and resource handling."
linkTitle: "13.13 Handling Configuration Changes"
categories:
- Android Development
- Kotlin Programming
- Mobile App Design
tags:
- Android
- Configuration Changes
- State Management
- Kotlin
- Resource Handling
date: 2024-11-17
type: docs
nav_weight: 14300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 13.13 Handling Configuration Changes

In the world of Android development, handling configuration changes is a critical aspect that ensures your application remains robust and user-friendly. Configuration changes occur when the state of the device changes, such as when the user rotates the screen, changes the language, or switches to a different display size. These changes can cause the Android system to destroy and recreate the activity, which can lead to loss of data and a poor user experience if not handled properly.

In this section, we will delve into the intricacies of managing configuration changes in Android applications using Kotlin. We will explore various strategies and best practices to maintain application state, manage resources efficiently, and provide a seamless user experience.

### Understanding Configuration Changes

Configuration changes in Android can be triggered by several factors, including:

- **Screen Rotation:** Changing the orientation of the device from portrait to landscape or vice versa.
- **Language Change:** Switching the device's language settings.
- **Keyboard Availability:** Connecting or disconnecting a hardware keyboard.
- **Screen Size:** Connecting to an external display or changing the device's display size.
- **Night Mode:** Switching between light and dark themes.

When a configuration change occurs, the Android system may destroy and recreate the activity to apply the new configuration. This process involves calling the `onDestroy()` method, followed by `onCreate()`, which can lead to loss of the activity's state if not managed correctly.

### Strategies for Handling Configuration Changes

To handle configuration changes effectively, developers can employ several strategies:

1. **Retaining Objects with Fragments:**
   - Use fragments to retain objects across configuration changes.
   - Implement the `setRetainInstance(true)` method in a fragment to retain its instance.

2. **Using ViewModel:**
   - Leverage the Android Architecture Components' ViewModel to store UI-related data.
   - ViewModel objects are designed to survive configuration changes.

3. **Persisting State with onSaveInstanceState():**
   - Override the `onSaveInstanceState()` method to save the activity's state.
   - Restore the state in the `onCreate()` method using the saved `Bundle`.

4. **Handling Configuration Changes Manually:**
   - Declare configuration changes in the manifest to prevent activity recreation.
   - Implement the `onConfigurationChanged()` method to handle changes manually.

5. **Using Jetpack Compose:**
   - Utilize Jetpack Compose's state management capabilities to handle configuration changes seamlessly.

Let's explore each of these strategies in detail.

### Retaining Objects with Fragments

Fragments provide a powerful mechanism to retain objects across configuration changes. By setting a fragment's instance to be retained, you can preserve its state and data even when the activity is recreated.

#### Implementing a Retained Fragment

Here's how you can implement a retained fragment in Kotlin:

```kotlin
class RetainedFragment : Fragment() {

    var someData: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Retain this fragment across configuration changes
        retainInstance = true
    }
}
```

In the above example, the `retainInstance` property is set to `true`, which ensures that the fragment's instance is retained across configuration changes. This allows you to store and retrieve data without losing it when the activity is recreated.

### Using ViewModel

The ViewModel class is part of the Android Architecture Components and is designed to store and manage UI-related data in a lifecycle-conscious way. ViewModels are retained during configuration changes, making them ideal for preserving UI state.

#### Creating a ViewModel

Here's an example of how to create and use a ViewModel in Kotlin:

```kotlin
class MyViewModel : ViewModel() {
    var someData: String? = null
}

// In your Activity or Fragment
class MyActivity : AppCompatActivity() {

    private lateinit var viewModel: MyViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Obtain the ViewModel
        viewModel = ViewModelProvider(this).get(MyViewModel::class.java)

        // Use the ViewModel to store and retrieve data
        viewModel.someData = "Hello, ViewModel!"
    }
}
```

In this example, the `MyViewModel` class extends `ViewModel`, and its instance is obtained using `ViewModelProvider`. The ViewModel can store data that survives configuration changes, ensuring that the UI state is preserved.

### Persisting State with onSaveInstanceState()

The `onSaveInstanceState()` method allows you to save the activity's state before it is destroyed. You can override this method to store data in a `Bundle`, which can be restored in the `onCreate()` method.

#### Saving and Restoring State

Here's how you can save and restore state using `onSaveInstanceState()`:

```kotlin
class MyActivity : AppCompatActivity() {

    private var someData: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Restore state if available
        someData = savedInstanceState?.getString("dataKey") ?: "Default Value"
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // Save state
        outState.putString("dataKey", someData)
    }
}
```

In this example, the `onSaveInstanceState()` method is used to save the `someData` variable in a `Bundle`. The state is then restored in the `onCreate()` method if available.

### Handling Configuration Changes Manually

In some cases, you may want to handle configuration changes manually to prevent the activity from being recreated. You can declare specific configuration changes in the manifest and implement the `onConfigurationChanged()` method.

#### Declaring Configuration Changes in the Manifest

Here's how you can declare configuration changes in the manifest:

```xml
<activity
    android:name=".MyActivity"
    android:configChanges="orientation|screenSize|keyboardHidden">
</activity>
```

By declaring configuration changes in the manifest, you instruct the Android system not to recreate the activity when these changes occur. Instead, the `onConfigurationChanged()` method is called.

#### Implementing onConfigurationChanged()

Here's an example of how to implement the `onConfigurationChanged()` method:

```kotlin
override fun onConfigurationChanged(newConfig: Configuration) {
    super.onConfigurationChanged(newConfig)

    // Handle configuration changes
    if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) {
        // Handle landscape orientation
    } else if (newConfig.orientation == Configuration.ORIENTATION_PORTRAIT) {
        // Handle portrait orientation
    }
}
```

In this example, the `onConfigurationChanged()` method is overridden to handle orientation changes manually. You can add logic to update the UI or perform other actions based on the new configuration.

### Using Jetpack Compose

Jetpack Compose is a modern UI toolkit for building native Android applications. It provides a declarative approach to UI development and includes powerful state management capabilities that simplify handling configuration changes.

#### State Management in Jetpack Compose

Jetpack Compose uses a concept called "state hoisting" to manage state. State hoisting involves moving state up to the composable's caller, allowing the caller to control the state.

Here's an example of state management in Jetpack Compose:

```kotlin
@Composable
fun MyComposable() {
    var text by remember { mutableStateOf("Hello, Compose!") }

    Column {
        Text(text = text)
        Button(onClick = { text = "Button Clicked!" }) {
            Text("Click Me")
        }
    }
}
```

In this example, the `remember` function is used to store the state of the `text` variable. The state is automatically preserved across configuration changes, ensuring that the UI remains consistent.

### Visualizing Configuration Change Handling

To better understand the flow of handling configuration changes, let's visualize the process using a flowchart.

```mermaid
flowchart TD
    A[Activity Created] --> B[Configuration Change Occurs]
    B --> C{Handle Configuration Change}
    C -->|Retain Fragment| D[Fragment Retained]
    C -->|Use ViewModel| E[ViewModel Retained]
    C -->|Save State| F[State Saved in Bundle]
    C -->|Handle Manually| G[Configuration Declared in Manifest]
    D --> H[Activity Recreated]
    E --> H
    F --> H
    G --> I[onConfigurationChanged() Called]
    I --> J[Update UI/State]
    H --> K[Restore State]
    J --> K
```

This flowchart illustrates the different strategies for handling configuration changes, including retaining fragments, using ViewModel, saving state in a bundle, and handling changes manually.

### Best Practices for Handling Configuration Changes

To ensure your application handles configuration changes effectively, consider the following best practices:

- **Use ViewModel for UI-related Data:** Leverage ViewModel to store and manage UI-related data that needs to survive configuration changes.
- **Persist Critical State:** Use `onSaveInstanceState()` to persist critical state information that cannot be easily recreated.
- **Avoid Large Objects in onSaveInstanceState():** Avoid storing large objects in the `Bundle` to prevent performance issues.
- **Handle Configuration Changes Manually When Necessary:** Declare configuration changes in the manifest and handle them manually when appropriate.
- **Utilize Jetpack Compose for Modern UI Development:** Use Jetpack Compose's state management capabilities to handle configuration changes seamlessly.

### Try It Yourself

Experiment with the code examples provided in this section. Try modifying the retained fragment example to store different types of data, or use the ViewModel example to manage more complex UI states. Explore Jetpack Compose's state management features by creating a simple composable that updates its state based on user interactions.

### Conclusion

Handling configuration changes is a crucial aspect of Android development that ensures your application provides a seamless user experience. By employing strategies such as retaining fragments, using ViewModel, persisting state with `onSaveInstanceState()`, and leveraging Jetpack Compose, you can effectively manage configuration changes and maintain application state.

Remember, this is just the beginning. As you continue to develop Android applications, you'll encounter more complex scenarios that require thoughtful handling of configuration changes. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is a configuration change in Android?

- [x] A change in device state that can cause the activity to be recreated.
- [ ] A change in the application's theme.
- [ ] A change in the user's location.
- [ ] A change in the network connectivity.

> **Explanation:** Configuration changes refer to changes in device state, such as screen rotation or language change, which can lead to activity recreation.

### Which method is used to save the activity's state before it is destroyed?

- [x] onSaveInstanceState()
- [ ] onDestroy()
- [ ] onPause()
- [ ] onStop()

> **Explanation:** The `onSaveInstanceState()` method is used to save the activity's state before it is destroyed.

### How can you retain a fragment across configuration changes?

- [x] Set retainInstance to true in the fragment.
- [ ] Use a static variable in the fragment.
- [ ] Override the onConfigurationChanged() method.
- [ ] Use a ViewModel.

> **Explanation:** Setting `retainInstance` to true in a fragment allows it to be retained across configuration changes.

### What is the purpose of the ViewModel class in Android?

- [x] To store and manage UI-related data in a lifecycle-conscious way.
- [ ] To handle network operations.
- [ ] To manage database connections.
- [ ] To provide animations for UI elements.

> **Explanation:** The ViewModel class is designed to store and manage UI-related data in a lifecycle-conscious way, surviving configuration changes.

### Which of the following is a best practice for handling configuration changes?

- [x] Use ViewModel for UI-related data.
- [ ] Store large objects in onSaveInstanceState().
- [ ] Avoid handling configuration changes manually.
- [ ] Use static variables to store data.

> **Explanation:** Using ViewModel for UI-related data is a best practice for handling configuration changes.

### What is the role of the onConfigurationChanged() method?

- [x] To handle configuration changes manually when declared in the manifest.
- [ ] To save the activity's state.
- [ ] To initialize the activity's UI.
- [ ] To start background services.

> **Explanation:** The `onConfigurationChanged()` method is used to handle configuration changes manually when they are declared in the manifest.

### How does Jetpack Compose help in handling configuration changes?

- [x] By providing state management capabilities that preserve state across changes.
- [ ] By automatically saving all UI data.
- [ ] By preventing configuration changes from occurring.
- [ ] By using static variables to store data.

> **Explanation:** Jetpack Compose provides state management capabilities that preserve state across configuration changes.

### What is state hoisting in Jetpack Compose?

- [x] Moving state up to the composable's caller to allow the caller to control the state.
- [ ] Storing state in a static variable.
- [ ] Using a ViewModel to manage state.
- [ ] Saving state in onSaveInstanceState().

> **Explanation:** State hoisting in Jetpack Compose involves moving state up to the composable's caller, allowing the caller to control the state.

### Which of the following is NOT a configuration change in Android?

- [x] A change in network connectivity.
- [ ] A change in screen orientation.
- [ ] A change in language settings.
- [ ] A change in screen size.

> **Explanation:** A change in network connectivity is not considered a configuration change in Android.

### True or False: Declaring configuration changes in the manifest prevents the activity from being recreated.

- [x] True
- [ ] False

> **Explanation:** Declaring configuration changes in the manifest prevents the activity from being recreated, allowing you to handle changes manually.

{{< /quizdown >}}
