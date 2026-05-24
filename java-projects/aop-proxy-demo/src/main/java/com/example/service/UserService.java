package com.example.service;

public interface UserService {
    String getUser(Long id);
    void createUser(String name);
}