#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", "; // Add a comma between elements but not after the last
                        // one
        }
    }
    os << "]";
    return os;
}
inline static const std::string roundDouble(const double input,
                                            const int decimal_places) {
    std::ostringstream str;
    str << std::fixed << std::setprecision(decimal_places);
    str << input;
    return str.str();
}
std::map<std::tuple<std::vector<int>, int>, int> initBonds(int L, int D) {
    std::map<std::tuple<std::vector<int>, int>, int> bonds;
    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        for (int d = 0; d < D; ++d) {
            bonds[std::make_tuple(indices, d)] = 1; // Initialize bond to +1
        }
    }
    return bonds;
}

std::map<std::vector<int>, int> initVertices(int L, int D) {
    std::map<std::vector<int>, int> vertices;
    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        vertices[indices] = 1; // Initialize spin to +1
    }
    return vertices;
}

// Function to initialize plaquettes using a map
std::map<std::tuple<std::vector<int>, int, int>, int> initPlaquettes(int L,
                                                                     int D) {
    std::map<std::tuple<std::vector<int>, int, int>, int> plaquettes;
    for (int i = 0; i < pow(L, D); ++i) {
        std::vector<int> indices(D, 0);
        int temp = i;
        for (int d = D - 1; d >= 0; --d) {
            indices[d] = temp % L;
            temp /= L;
        }
        for (int d1 = 0; d1 < D; ++d1) {
            for (int d2 = d1 + 1; d2 < D; ++d2) {
                plaquettes[std::make_tuple(indices, d1, d2)] =
                    1; // Initialize spin to +1
            }
        }
    }
    return plaquettes;
}
// Helper function to convert multi-dimensional indices to a flat index
int getFlatIndex(const std::vector<int> &indices, int L) {
    int flatIndex = 0;
    int multiplier = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        flatIndex += indices[i] * multiplier;
        multiplier *= L;
    }
    return flatIndex;
}

void applyError(
    int L, std::map<std::vector<int>, int> &vertices,
    std::map<std::tuple<std::vector<int>, int, int>, int> &plaquettes,
    std::map<std::tuple<std::vector<int>, int>, int> &bonds,
    const std::vector<int> &vindex, int bond, double p, std::mt19937 &gen,
    int D) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> errorTypeDis(
        0, 2); // 0 for X, 1 for Z, 2 for Y; only Z if D = 1

    if (dis(gen) < p) {
        int errorType;
        if (D > 1) {
            errorType = errorTypeDis(gen);
        } else {
            errorType = 3;
        }
        if (p == 0) {
            std::cout << "WHAT?" << std::endl;
        }

        // Hard code error case
        errorType = 0;

        // Calculate the second vertex connected to the bond
        std::vector<int> secondVertex = vindex;
        secondVertex[bond] = (secondVertex[bond] + 1) % L;

        switch (errorType) {
        case 0:
            // Flip all faces attached to the bond
            for (int j = 0; j < vindex.size(); j++) {
                if (j != bond) {
                    auto faceTuple = std::make_tuple(vindex, std::min(bond, j),
                                                     std::max(bond, j));
                    plaquettes[faceTuple] *= -1;
                    std::vector<int> altVindex = vindex;
                    altVindex[j] = (altVindex[j] - 1 + L) % L;
                    auto altFaceTuple = std::make_tuple(
                        altVindex, std::min(bond, j), std::max(bond, j));
                    plaquettes[altFaceTuple] *= -1;
                }
            }
            break;

        case 1:
            // Flip both vertices attached to the bond
            vertices[vindex] *= -1;
            vertices[secondVertex] *= -1;
            break;

        case 2:
            // Flip all faces attached to the bond
            for (int j = 0; j < vindex.size(); j++) {
                if (j != bond) {
                    auto faceTuple = std::make_tuple(vindex, std::min(bond, j),
                                                     std::max(bond, j));
                    plaquettes[faceTuple] *= -1;
                    std::vector<int> altVindex = vindex;
                    altVindex[j] = (altVindex[j] - 1 + L) % L;
                    auto altFaceTuple = std::make_tuple(
                        altVindex, std::min(bond, j), std::max(bond, j));
                    plaquettes[altFaceTuple] *= -1;
                }
            }
            // Flip both vertices attached to the bond
            vertices[vindex] *= -1;
            vertices[secondVertex] *= -1;
            break;
        case 3:
            bonds[std::make_tuple(vindex, bond)] *= -1;
            bonds[std::make_tuple(secondVertex, bond)] *= -1;
        }
    }
}
int main(int argc, char *argv[]) {
    int L = std::stoi(argv[1]);
    int Lsmall = std::stoi(argv[2]);
    int D = std::stoi(argv[3]); // New parameter for dimensions
    int numSamples = std::stoi(argv[4]);
    std::random_device rd;
    int pNum = 0;

#pragma omp parallel for
    for (int pi = 0; pi < 20; pi++) {
        double p = pi * 0.01;
        std::ostringstream filename;
        filename << "traindata/toric/d=" << D << "/measurements_L=" << Lsmall
                 << "_p=" << roundDouble(p, 3) << ".csv";
        std::ofstream fileObj(filename.str(), std::ios::out | std::ios::trunc);

        std::vector<std::string> buffer;
        const int bufferSize = 1000;

        std::mt19937 gen(rd() + pNum);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int n = 0; n < numSamples; n++) {
            // Initialize vertices and plaquettes
            auto vertices = initVertices(L, D);
            auto plaquettes = initPlaquettes(L, D);
            auto bonds = initBonds(L, D);

            // Iterate over each vertex in the vertex map
            for (const auto &vertex : vertices) {
                const std::vector<int> &vindex = vertex.first;
                // Iterate over all the bonds attached to a vertex
                for (int i = 0; i < D; ++i) {
                    applyError(L, vertices, plaquettes, bonds, vindex, i, p,
                               gen, D);
                }
            }

            std::ostringstream line;
            // Write plaquettes to the line only if D > 1
            if (D == 2) {
                for (int vi = 0; vi < Lsmall; vi++) {
                    for (int vj = 0; vj < Lsmall; vj++) {
                        std::vector<int> vertex = {vi, vj};

                        std::tuple<std::vector<int>, int, int> indexer(vertex,
                                                                       0, 1);
                        line << plaquettes[indexer] << " ";
                    }
                }
                // // Write vertices to the line
                // for (const auto &vertex : vertices) {
                //     line << vertex.second << " ";
                // }
            } else {
                for (const auto &bond : bonds) {
                    line << bond.second << " ";
                }
            }

            buffer.push_back(line.str());

            if (buffer.size() >= bufferSize) {
                for (const auto &l : buffer)
                    fileObj << l << '\n';
                buffer.clear();
            }
        }
        pNum += 1;
        for (const auto &l : buffer)
            fileObj << l << '\n';
        buffer.clear();
        fileObj.close();
    }
    return 0;
}
