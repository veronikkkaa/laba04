#include "tfidf.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

std::string TFIDFEngine::normalize(const std::string& word) {
    std::string result;
    result.reserve(word.size());
    std::transform(word.begin(), word.end(), std::back_inserter(result),
                   [](unsigned char c) { return std::tolower(c); });
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](unsigned char c) { return !std::isalnum(c); }),
                 result.end());
    return result;
}

std::vector<std::string> TFIDFEngine::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string raw;
    while (iss >> raw) {
        std::string token = normalize(raw);
        if (!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

void TFIDFEngine::loadDocuments(const std::string& listFile) {
    std::ifstream list(listFile);
    if (!list.is_open()) {
        std::cerr << "Error: cannot open " << listFile << "\n";
        return;
    }

    std::string filename;
    while (std::getline(list, filename)) {
        if (!filename.empty() && filename.back() == '\r')
            filename.pop_back();
        if (filename.empty()) continue;

        std::ifstream docFile(filename);
        if (!docFile.is_open()) {
            std::cerr << "Warning: cannot open document " << filename << "\n";
            continue;
        }

        Document doc;
        doc.name = filename;

        std::string line;
        while (std::getline(docFile, line)) {
            auto tokens = tokenize(line);
            for (const auto& t : tokens) {
                doc.words.push_back(t);
                doc.wordCount[t]++;
            }
        }
        docs.push_back(std::move(doc));
    }

    std::cout << "Loaded " << docs.size() << " document(s).\n\n";
}

double TFIDFEngine::tf(const std::string& word, const Document& doc) const {
    if (doc.words.empty()) return 0.0;
    auto it = doc.wordCount.find(word);
    int count = (it != doc.wordCount.end()) ? it->second : 0;
    return static_cast<double>(count) / static_cast<double>(doc.words.size());
}

int TFIDFEngine::df(const std::string& word) const {
    int count = 0;
    for (const auto& doc : docs)
        if (doc.wordCount.count(word)) ++count;
    return count;
}

double TFIDFEngine::idf(const std::string& word) const {
    int d = df(word);
    if (d == 0) return 0.0;
    return std::log(static_cast<double>(docs.size()) / static_cast<double>(d));
}

double TFIDFEngine::tfidf(const std::string& word, const Document& doc) const {
    return tf(word, doc) * idf(word);
}

std::vector<std::string> TFIDFEngine::docsWithWord(const std::string& word) const {
    std::vector<std::string> result;
    for (const auto& doc : docs)
        if (doc.wordCount.count(word))
            result.push_back(doc.name);
    return result;
}

const Document* TFIDFEngine::findDoc(const std::string& name) const {
    for (const auto& doc : docs)
        if (doc.name == name) return &doc;
    return nullptr;
}

void TFIDFEngine::handleWord(const std::string& word) const {
    auto names = docsWithWord(word);
    std::cout << "Word: " << word << "\n"
              << "Documents total: " << docs.size() << "\n"
              << "Documents with word: " << names.size() << "\n"
              << std::fixed << std::setprecision(4)
              << "IDF: " << idf(word) << "\n"
              << "Appears in:\n";
    for (const auto& n : names)
        std::cout << "  - " << n << "\n";
}

void TFIDFEngine::handleWordInDoc(const std::string& word, const std::string& docName) const {
    const Document* doc = findDoc(docName);
    if (!doc) {
        std::cout << "Error: document \"" << docName << "\" not found.\n";
        return;
    }
    auto it = doc->wordCount.find(word);
    int count = (it != doc->wordCount.end()) ? it->second : 0;

    std::cout << "Word: " << word << "\n"
              << "Document: " << docName << "\n"
              << "Count: " << count << "\n"
              << std::fixed << std::setprecision(4)
              << "TF: " << tf(word, *doc) << "\n"
              << "TF-IDF: " << tfidf(word, *doc) << "\n";
}

void TFIDFEngine::handleDoc(const std::string& docName) const {
    const Document* doc = findDoc(docName);
    if (!doc) {
        std::cout << "Error: document \"" << docName << "\" not found.\n";
        return;
    }

    std::set<std::string> unique(doc->words.begin(), doc->words.end());

    std::vector<std::pair<std::string, double>> scores;
    scores.reserve(unique.size());
    for (const auto& w : unique)
        scores.emplace_back(w, tfidf(w, *doc));

    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "Document: " << docName << "\n"
              << "Total words: " << doc->words.size() << "\n"
              << "Unique words: " << unique.size() << "\n"
              << "Top words:\n";

    int limit = std::min(5, static_cast<int>(scores.size()));
    for (int i = 0; i < limit; ++i)
        std::cout << std::fixed << std::setprecision(4)
                  << "  " << (i + 1) << ". " << scores[i].first
                  << " (" << scores[i].second << ")\n";
}

void TFIDFEngine::handleQuery(const std::vector<std::string>& queryWords) const {
    std::vector<std::pair<std::string, double>> scores;
    for (const auto& doc : docs) {
        double score = 0.0;
        for (const auto& w : queryWords)
            score += tfidf(w, doc);
        if (score > 0.0)
            scores.emplace_back(doc.name, score);
    }

    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "Query:";
    for (const auto& w : queryWords) std::cout << " " << w;
    std::cout << "\nResults:\n";

    if (scores.empty()) {
        std::cout << "  No results found.\n";
        return;
    }
    for (int i = 0; i < static_cast<int>(scores.size()); ++i)
        std::cout << std::fixed << std::setprecision(4)
                  << "  " << (i + 1) << ". " << scores[i].first
                  << " (" << scores[i].second << ")\n";
}

void TFIDFEngine::processQuery(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    std::transform(cmd.begin(), cmd.end(), cmd.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (cmd == "WORD") {
        std::string word;
        if (!(iss >> word)) { std::cout << "Error: WORD requires an argument.\n"; return; }
        handleWord(normalize(word));

    } else if (cmd == "WORD_IN_DOC") {
        std::string word, docName;
        if (!(iss >> word >> docName)) {
            std::cout << "Error: WORD_IN_DOC requires <word> <document>.\n"; return;
        }
        handleWordInDoc(normalize(word), docName);

    } else if (cmd == "DOC") {
        std::string docName;
        if (!(iss >> docName)) { std::cout << "Error: DOC requires a document name.\n"; return; }
        handleDoc(docName);

    } else if (cmd == "QUERY") {
        std::vector<std::string> words;
        std::string w;
        while (iss >> w) words.push_back(normalize(w));
        if (words.empty()) { std::cout << "Error: QUERY requires at least one word.\n"; return; }
        handleQuery(words);

    } else {
        std::cout << "Error: unknown command \"" << cmd << "\".\n"
                  << "Available: WORD, WORD_IN_DOC, DOC, QUERY\n";
    }
}
