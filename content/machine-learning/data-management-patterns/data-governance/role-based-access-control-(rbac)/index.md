---
linkTitle: "Role-Based Access Control"
title: "Role-Based Access Control: Controlling Data Access Based on User Roles"
description: "Implementing Role-Based Access Control (RBAC) to manage data access based on user roles within an organization to ensure data governance and security."
categories:
- Data Management Patterns
tags:
- Data Governance
- Security
- Access Control
- Data Management
- Organization
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-governance/role-based-access-control-(rbac)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Role-Based Access Control (RBAC) is a method used to restrict access to data and system resources based on the roles assigned to individual users within an organization. By associating specific roles with corresponding permissions, RBAC ensures that users gain access only to the data and functionalities necessary for their job functions, thus enhancing security and adherence to compliance regulations.

## Benefits of RBAC
1. **Improved Security:** Minimizes the risk of unauthorized access.
2. **Operational Efficiency:** Simplifies the management of user permissions.
3. **Regulatory Compliance:** Helps in meeting data governance and regulatory requirements, such as GDPR and HIPAA.
4. **Scalability:** Easily adjustable as the organization grows and user roles evolve.

## Key Components of RBAC
1. **User:** An individual with access to the computer system.
2. **Role:** A job function or title which defines the user's authority and responsibility.
3. **Permission:** The authorization to perform specific operations or access certain data.
4. **Session:** A mapping between a user and activated roles.

## Detailed Example

Consider a healthcare organization where different roles such as Doctor, Nurse, and Administrator have different levels of access to the patient records.

### Example in Python with Flask

```python
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

users = {
    'alice': {'password': 'password123', 'role': 'doctor'},
    'bob': {'password': 'password456', 'role': 'nurse'},
    'charlie': {'password': 'password789', 'role': 'admin'}
}

permissions = {
    'doctor': ['view_patient_records', 'edit_patient_records'],
    'nurse': ['view_patient_records'],
    'admin': ['view_patient_records', 'edit_patient_records', 'add_user']
}

def check_auth(username, password):
    user = users.get(username)
    return user and user['password'] == password

def requires_permission(permission):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth = request.authorization
            if not auth or not check_auth(auth.username, auth.password):
                return jsonify({'message': 'Unauthorized'}), 403
            user_role = users[auth.username]['role']
            if permission not in permissions[user_role]:
                return jsonify({'message': 'Permission Denied'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/view_patient_records')
@requires_permission('view_patient_records')
def view_patient_records():
    return jsonify({'records': 'This is the patient records.'})

@app.route('/edit_patient_records')
@requires_permission('edit_patient_records')
def edit_patient_records():
    return jsonify({'message': 'Edit patient records page.'})

@app.route('/add_user')
@requires_permission('add_user')
def add_user():
    return jsonify({'message': 'Add user page.'})

if __name__ == "__main__":
    app.run(debug=True)
```

### Example in JavaScript with Node.js and Express

```javascript
const express = require('express');
const app = express();

// Sample data (usually in a database)
const users = {
    alice: { password: 'password123', role: 'doctor' },
    bob: { password: 'password456', role: 'nurse' },
    charlie: { password: 'password789', role: 'admin' }
};

// Permissions object
const permissions = {
    doctor: ['view_patient_records', 'edit_patient_records'],
    nurse: ['view_patient_records'],
    admin: ['view_patient_records', 'edit_patient_records', 'add_user']
};

// Middleware to check authentication
const authenticate = (req, res, next) => {
    const { username, password } = req.headers;
    if (!username || !password || !users[username] || users[username].password !== password) {
        return res.status(403).json({ message: 'Unauthorized' });
    }
    req.user = users[username];
    next();
};

// Middleware for RBAC permission check
const requiresPermission = (permission) => (req, res, next) => {
    const role = req.user.role;
    if (!permissions[role].includes(permission)) {
        return res.status(403).json({ message: 'Permission Denied' });
    }
    next();
};

app.use(authenticate);

app.get('/view_patient_records', requiresPermission('view_patient_records'), (req, res) => {
    res.json({ records: 'This is the patient records.' });
});

app.get('/edit_patient_records', requiresPermission('edit_patient_records'), (req, res) => {
    res.json({ message: 'Edit patient records page.' });
});

app.post('/add_user', requiresPermission('add_user'), (req, res) => {
    res.json({ message: 'Add user page.' });
});

app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
```

## Related Design Patterns

1. **Attribute-Based Access Control (ABAC):**
   - Instead of roles, ABAC uses attributes (such as user, resource, and environment characteristics) to define access.
   - Example: A policy granting access based on time of day or geographical location.

2. **Mandatory Access Control (MAC):**
   - Uses a strict policy determined by a central authority rather than individual user discretion.
   - Example: Government and military systems where classifications (such as Top Secret and Confidential) are used.

3. **Discretionary Access Control (DAC):**
   - The owner of the protected system or resource sets policies and allows grant access to others.
   - Example: Sharing a folder in a computer system where the owner sets permissions.

## Additional Resources

- NIST RBAC Standard: [NIST RBAC](https://csrc.nist.gov/projects/role-based-access-control)
- OWASP Access Control Cheat Sheet: [OWASP Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- Flask Documentation: [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- Express Documentation: [Express](http://expressjs.com/)

## Summary

Role-Based Access Control (RBAC) is a pivotal design pattern in ensuring data security and governance within an organization. By assigning roles with specific permissions, organizations can control data access effectively, ensure compliance with regulations, and streamline the administration of user privileges. This pattern is versatile and can be implemented using various programming languages and frameworks, offering both flexibility and robustness for modern software architectures.
