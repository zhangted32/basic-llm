package com.example.repository;

import org.springframework.stereotype.Repository;

import java.util.HashMap;
import java.util.Map;

@Repository
public class UserRepository {

    private final Map<Long, String> users = new HashMap<>();
    private long nextId = 1;

    public String findById(Long id) {
        if (!users.containsKey(id)) {
            throw new RuntimeException("User not found with id: " + id);
        }
        return users.get(id);
    }

    public Long save(String name) {
        Long id = nextId++;
        users.put(id, name);
        return id;
    }

    public void delete(Long id) {
        users.remove(id);
    }
}