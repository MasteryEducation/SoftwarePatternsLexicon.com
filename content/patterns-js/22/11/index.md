---
canonical: "https://softwarepatternslexicon.com/patterns-js/22/11"
title: "Progressive Web Apps (PWAs) and Future Web APIs"
description: "Explore the advancements in Progressive Web Apps and discover upcoming web APIs that enhance web application capabilities."
linkTitle: "22.11 Progressive Web Apps (PWAs) and Future Web APIs"
tags:
- "Progressive Web Apps"
- "PWAs"
- "Web APIs"
- "JavaScript"
- "Web Development"
- "Payment Request API"
- "Web NFC API"
- "Web Bluetooth API"
date: 2024-11-25
type: docs
nav_weight: 231000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 22.11 Progressive Web Apps (PWAs) and Future Web APIs

Progressive Web Apps (PWAs) have revolutionized the way we think about web applications by combining the best of web and mobile apps. They offer a seamless, fast, and engaging user experience that can work offline and be installed on a user's device. In this section, we will delve into the core features of PWAs and explore some of the latest and upcoming web APIs that promise to further enhance the capabilities of web applications.

### Understanding Progressive Web Apps (PWAs)

**Progressive Web Apps** are web applications that use modern web capabilities to deliver an app-like experience to users. They are built using standard web technologies including HTML, CSS, and JavaScript, but they offer a level of functionality that was traditionally only available to native apps. Let's recap the key features that make PWAs stand out:

- **Responsive**: PWAs are designed to work on any device, regardless of screen size or orientation.
- **Connectivity Independent**: They can function offline or on low-quality networks using service workers.
- **App-like Interactions**: PWAs provide an app-like experience with smooth animations and interactions.
- **Fresh**: They are always up-to-date thanks to the service worker update process.
- **Safe**: PWAs are served via HTTPS to prevent snooping and ensure content integrity.
- **Discoverable**: They are identifiable as applications thanks to W3C manifests and service worker registration.
- **Re-engageable**: Features like push notifications make it easy to re-engage users.
- **Installable**: Users can add them to their home screen without the need for an app store.
- **Linkable**: They can be shared easily via URL and do not require complex installation.

### Future Web APIs Enhancing PWAs

As web technologies evolve, new APIs are being introduced to extend the capabilities of web applications. These APIs allow developers to create more powerful and feature-rich PWAs. Below, we explore some of the most promising upcoming web APIs.

#### Payment Request API

The [Payment Request API](https://developer.mozilla.org/en-US/docs/Web/API/Payment_Request_API) is designed to simplify the payment process on the web. It provides a consistent user experience across different browsers and devices, reducing the friction of online payments.

**Example Usage:**

```javascript
if (window.PaymentRequest) {
    const supportedInstruments = [{
        supportedMethods: 'basic-card',
        data: {
            supportedNetworks: ['visa', 'mastercard']
        }
    }];

    const details = {
        total: {
            label: 'Total',
            amount: { currency: 'USD', value: '55.00' }
        }
    };

    const request = new PaymentRequest(supportedInstruments, details);

    request.show().then(paymentResponse => {
        // Process paymentResponse here
        return paymentResponse.complete('success');
    }).catch(err => {
        console.error('Payment failed', err);
    });
} else {
    console.log('Payment Request API not supported');
}
```

**Enhancements for PWAs:**

- **Streamlined Checkout**: Reduces the steps required to complete a purchase, improving conversion rates.
- **Consistent Experience**: Provides a uniform payment interface across different platforms and devices.

#### Web NFC API

The [Web NFC API](https://developer.mozilla.org/en-US/docs/Web/API/Web_NFC_API) allows web applications to read and write to NFC tags. This API is particularly useful for applications that require interaction with physical objects, such as inventory management or interactive exhibits.

**Example Usage:**

```javascript
if ('NDEFReader' in window) {
    const ndef = new NDEFReader();
    ndef.scan().then(() => {
        console.log("Scan started successfully.");
        ndef.onreading = event => {
            const decoder = new TextDecoder();
            for (const record of event.message.records) {
                console.log("Record type:  " + record.recordType);
                console.log("MIME type:    " + record.mediaType);
                console.log("=== data ===\n" + decoder.decode(record.data));
            }
        };
    }).catch(error => {
        console.log(`Error! Scan failed to start: ${error}.`);
    });
} else {
    console.log('Web NFC API not supported');
}
```

**Enhancements for PWAs:**

- **Physical Interaction**: Enables seamless interaction with the physical world, enhancing user engagement.
- **Innovative Use Cases**: Opens up possibilities for creative applications in retail, museums, and more.

#### Web Bluetooth API

The [Web Bluetooth API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Bluetooth_API) allows web applications to connect to Bluetooth devices directly. This API is ideal for applications that need to interact with IoT devices, such as fitness trackers or smart home devices.

**Example Usage:**

```javascript
navigator.bluetooth.requestDevice({
    filters: [{ services: ['heart_rate'] }]
})
.then(device => {
    console.log('Connecting to GATT Server...');
    return device.gatt.connect();
})
.then(server => {
    console.log('Getting Heart Rate Service...');
    return server.getPrimaryService('heart_rate');
})
.then(service => {
    console.log('Getting Heart Rate Measurement Characteristic...');
    return service.getCharacteristic('heart_rate_measurement');
})
.then(characteristic => {
    return characteristic.startNotifications().then(_ => {
        console.log('> Notifications started');
        characteristic.addEventListener('characteristicvaluechanged',
            handleHeartRateMeasurement);
    });
})
.catch(error => {
    console.log('Argh! ' + error);
});

function handleHeartRateMeasurement(event) {
    let value = event.target.value;
    let heartRate = value.getUint8(1);
    console.log('Heart Rate: ' + heartRate);
}
```

**Enhancements for PWAs:**

- **IoT Integration**: Facilitates the creation of web applications that can interact with a wide range of Bluetooth-enabled devices.
- **Real-Time Data**: Enables real-time data collection and interaction, enhancing user experience.

#### File System Access API

The [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API) provides a way for web applications to read and write files on the user's local file system. This API is particularly useful for applications that require file manipulation, such as text editors or image processors.

**Example Usage:**

```javascript
async function getFile() {
    // Prompt user to select a file
    const [fileHandle] = await window.showOpenFilePicker();
    const file = await fileHandle.getFile();
    const contents = await file.text();
    console.log(contents);
}

async function saveFile() {
    const options = {
        types: [{
            description: 'Text Files',
            accept: {'text/plain': ['.txt']},
        }],
    };
    const handle = await window.showSaveFilePicker(options);
    const writable = await handle.createWritable();
    await writable.write('Hello, world!');
    await writable.close();
}
```

**Enhancements for PWAs:**

- **Local File Access**: Allows web applications to offer functionality similar to native applications, such as saving and loading files.
- **Enhanced User Control**: Provides users with more control over their files and data.

### Considerations for Browser Support and Fallbacks

While these APIs offer exciting new possibilities, it's important to consider browser support and provide fallbacks for unsupported environments. Here are some strategies to ensure a smooth user experience:

- **Feature Detection**: Use feature detection to check if an API is supported before using it. This can be done using simple conditional checks, as shown in the examples above.
- **Progressive Enhancement**: Build your application with a basic level of functionality that works across all browsers, and enhance it with advanced features where supported.
- **Polyfills**: Consider using polyfills to emulate the behavior of newer APIs in older browsers. This can help bridge the gap until full support is available.
- **User Education**: Inform users about the benefits of using a modern browser and provide guidance on how to upgrade if necessary.

### Encouraging Experimentation

As developers, it's important to stay curious and experiment with new technologies. Here are some ways to get started with these APIs:

- **Build a Demo**: Create a small project that uses one or more of these APIs to solve a real-world problem or demonstrate a concept.
- **Contribute to Open Source**: Many open-source projects are exploring these APIs. Contributing to such projects can be a great way to learn and make an impact.
- **Share Your Knowledge**: Write blog posts, create tutorials, or give talks about your experiences with these APIs. Sharing knowledge helps the community grow and innovate.

### Conclusion

Progressive Web Apps and the latest web APIs are transforming the web development landscape. By leveraging these technologies, we can create powerful, engaging, and versatile web applications that rival native apps in functionality and user experience. Remember, this is just the beginning. As you progress, you'll build more complex and interactive web applications. Keep experimenting, stay curious, and enjoy the journey!

## Test Your Knowledge on Progressive Web Apps and Future Web APIs

{{< quizdown >}}

### What is a key feature of Progressive Web Apps (PWAs)?

- [x] They can work offline.
- [ ] They require an app store for installation.
- [ ] They are only available on mobile devices.
- [ ] They cannot be updated automatically.

> **Explanation:** PWAs can work offline using service workers, which is one of their key features.

### Which API is used to simplify the payment process on the web?

- [x] Payment Request API
- [ ] Web NFC API
- [ ] Web Bluetooth API
- [ ] File System Access API

> **Explanation:** The Payment Request API is designed to simplify the payment process on the web.

### What does the Web NFC API allow web applications to do?

- [x] Read and write to NFC tags.
- [ ] Connect to Bluetooth devices.
- [ ] Access the local file system.
- [ ] Simplify online payments.

> **Explanation:** The Web NFC API allows web applications to read and write to NFC tags.

### Which API enables web applications to connect to Bluetooth devices?

- [x] Web Bluetooth API
- [ ] Payment Request API
- [ ] Web NFC API
- [ ] File System Access API

> **Explanation:** The Web Bluetooth API allows web applications to connect to Bluetooth devices.

### What is the purpose of the File System Access API?

- [x] To read and write files on the user's local file system.
- [ ] To connect to NFC tags.
- [ ] To simplify online payments.
- [ ] To connect to Bluetooth devices.

> **Explanation:** The File System Access API provides a way for web applications to read and write files on the user's local file system.

### What strategy should be used to ensure a smooth user experience when using new web APIs?

- [x] Progressive Enhancement
- [ ] Only support modern browsers
- [ ] Ignore unsupported browsers
- [ ] Use deprecated APIs

> **Explanation:** Progressive Enhancement ensures a smooth user experience by building a basic level of functionality that works across all browsers and enhancing it with advanced features where supported.

### What is a benefit of using the Payment Request API in PWAs?

- [x] Streamlined checkout process
- [ ] Access to NFC tags
- [ ] Connection to Bluetooth devices
- [ ] Local file access

> **Explanation:** The Payment Request API streamlines the checkout process, improving conversion rates.

### How can developers handle unsupported web APIs in older browsers?

- [x] Use polyfills
- [ ] Ignore older browsers
- [ ] Use only deprecated APIs
- [ ] Avoid using new APIs

> **Explanation:** Polyfills can be used to emulate the behavior of newer APIs in older browsers.

### What is a recommended way to start experimenting with new web APIs?

- [x] Build a demo project
- [ ] Wait for full browser support
- [ ] Avoid using them until they are stable
- [ ] Only use them in production

> **Explanation:** Building a demo project is a great way to start experimenting with new web APIs.

### True or False: PWAs can be installed on a user's device without an app store.

- [x] True
- [ ] False

> **Explanation:** PWAs can be installed on a user's device directly from the browser without the need for an app store.

{{< /quizdown >}}
