#!/usr/bin/env python3
"""Test script to verify multiple nodes generation."""

from src.model.qwen_mlx import QwenMLXInterface

model = QwenMLXInterface()
model.load_model()

trace = """java.lang.RuntimeException: Database connection failed
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:28)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)
Caused by: java.sql.SQLException: Connection refused
    at com.example.repository.DatabaseRepository.connect(DatabaseRepository.java:56)
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:40)
    ... 5 more"""

ecg = model.generate_ecg(trace, {})

print('=' * 60)
print('ECG RESULTS')
print('=' * 60)
print(f'Nodes generated: {len(ecg.get("nodes", []))}')
print(f'Edges generated: {len(ecg.get("edges", []))}')
print()
print('Nodes:')
for node in ecg.get('nodes', []):
    print(f'  - {node.get("id")}')
