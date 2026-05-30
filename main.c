#include "tfidf.h"
#include <iostream>
#include <string>

int main() {
    TFIDFEngine engine;
    engine.loadDocuments("documents.txt");

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        engine.processQuery(line);
        std::cout << "\n";
    }

    return 0;
}
