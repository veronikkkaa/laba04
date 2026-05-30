#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

struct Document {
    std::string name;
    std::vector<std::string> words;
    std::unordered_map<std::string, int> wordCount;
};

class TFIDFEngine {
public:
    void loadDocuments(const std::string& listFile);
    void processQuery(const std::string& line);

private:
    std::vector<Document> docs;

    static std::string normalize(const std::string& word);
    static std::vector<std::string> tokenize(const std::string& text);

    double tf(const std::string& word, const Document& doc) const;
    double idf(const std::string& word) const;
    double tfidf(const std::string& word, const Document& doc) const;

    int df(const std::string& word) const;
    std::vector<std::string> docsWithWord(const std::string& word) const;

    void handleWord(const std::string& word) const;
    void handleWordInDoc(const std::string& word, const std::string& docName) const;
    void handleDoc(const std::string& docName) const;
    void handleQuery(const std::vector<std::string>& queryWords) const;

    const Document* findDoc(const std::string& name) const;
};
